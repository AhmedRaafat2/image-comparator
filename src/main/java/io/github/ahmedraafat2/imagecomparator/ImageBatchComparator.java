package io.github.ahmedraafat2.imagecomparator;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * Compares image collections in two directory trees by relative path and file name.
 * Processing is sequential to keep memory usage predictable.
 */
public final class ImageBatchComparator {
    private static final Set<String> IMAGE_EXTENSIONS =
            Set.of("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp");

    private final ImageComparator imageComparator;

    public ImageBatchComparator() {
        this(new ImageComparator());
    }

    ImageBatchComparator(ImageComparator imageComparator) {
        this.imageComparator = Objects.requireNonNull(imageComparator, "imageComparator");
    }

    /** Compares directories using the default pixel and size policies. */
    public BatchComparisonResult compareDirectories(Path referenceDirectory, Path actualDirectory,
                                                     Path reportsDirectory) throws IOException {
        return compareDirectories(referenceDirectory, actualDirectory, reportsDirectory,
                ImageComparator.ComparisonOptions.defaults());
    }

    /**
     * Recursively compares supported images in the reference and actual directories.
     * Reports are written under {@code reportsDirectory}; the directory must not be inside either input tree.
     *
     * @param referenceDirectory directory containing baseline images
     * @param actualDirectory directory containing images to compare
     * @param reportsDirectory destination for diagnostic images and {@code summary.json}
     * @param options pixel thresholds and size mismatch policy
     * @return one result per unique relative image name
     * @throws IOException if an input directory cannot be read or the report summary cannot be written
     * @throws IllegalArgumentException if required paths are null or the directories overlap
     */
    public BatchComparisonResult compareDirectories(Path referenceDirectory, Path actualDirectory,
                                                     Path reportsDirectory,
                                                     ImageComparator.ComparisonOptions options) throws IOException {
        Objects.requireNonNull(referenceDirectory, "referenceDirectory");
        Objects.requireNonNull(actualDirectory, "actualDirectory");
        Objects.requireNonNull(reportsDirectory, "reportsDirectory");
        Objects.requireNonNull(options, "options");

        Path referenceRoot = requireDirectory(referenceDirectory, "referenceDirectory");
        Path actualRoot = requireDirectory(actualDirectory, "actualDirectory");
        if (referenceRoot.startsWith(actualRoot) || actualRoot.startsWith(referenceRoot)) {
            throw new IllegalArgumentException("Reference and actual directories must not overlap");
        }

        Path reportsAbsolute = reportsDirectory.toAbsolutePath().normalize();
        Path reportsResolved = resolveIncludingMissingPath(reportsAbsolute);
        if (reportsResolved.startsWith(referenceRoot) || reportsResolved.startsWith(actualRoot)) {
            throw new IllegalArgumentException("Reports directory must not be inside either input directory");
        }
        Files.createDirectories(reportsAbsolute);

        Map<String, List<ImageFile>> referenceFiles = discover(referenceRoot);
        Map<String, List<ImageFile>> actualFiles = discover(actualRoot);
        TreeSet<String> keys = new TreeSet<>();
        keys.addAll(referenceFiles.keySet());
        keys.addAll(actualFiles.keySet());

        List<BatchComparisonResult.FileResult> results = new ArrayList<>(keys.size());
        for (String key : keys) {
            List<ImageFile> references = referenceFiles.getOrDefault(key, List.of());
            List<ImageFile> actuals = actualFiles.getOrDefault(key, List.of());
            results.add(compareEntry(key, references, actuals, referenceRoot, actualRoot,
                    reportsAbsolute, options));
        }

        Path summaryFile = reportsAbsolute.resolve("summary.json");
        BatchComparisonResult result = new BatchComparisonResult(results, summaryFile);
        Files.writeString(summaryFile, toJson(result, reportsAbsolute), StandardCharsets.UTF_8);
        return result;
    }

    private BatchComparisonResult.FileResult compareEntry(
            String key, List<ImageFile> references, List<ImageFile> actuals,
            Path referenceRoot, Path actualRoot, Path reportsRoot,
            ImageComparator.ComparisonOptions options) {
        ImageFile reference = references.isEmpty() ? null : references.get(0);
        ImageFile actual = actuals.isEmpty() ? null : actuals.get(0);
        ImageFile representative = reference != null ? reference : actual;
        Path relativePath = representative.relativePath;

        if (references.size() > 1 || actuals.size() > 1) {
            return fileResult(relativePath, BatchComparisonResult.Status.AMBIGUOUS,
                    reference == null ? null : reference.path,
                    actual == null ? null : actual.path,
                    null, null, null,
                    "More than one image has this relative name without an extension: " + key);
        }

        if (reference != null && actual != null) {
            if (!reference.extension.equals(actual.extension)) {
                return fileResult(relativePath, BatchComparisonResult.Status.EXTENSION_MISMATCH,
                        reference.path, actual.path, null, null, null,
                        "Image extensions differ: reference=" + reference.extension
                                + ", actual=" + actual.extension);
            }
            return comparePair(relativePath, reference.path, actual.path, reportsRoot, options,
                    BatchComparisonResult.Status.DIFFERENT);
        }

        Path referencePath = reference != null
                ? reference.path : referenceRoot.resolve(representative.relativePath);
        Path actualPath = actual != null
                ? actual.path : actualRoot.resolve(representative.relativePath);
        BatchComparisonResult.Status missingStatus = reference == null
                ? BatchComparisonResult.Status.MISSING_REFERENCE
                : BatchComparisonResult.Status.MISSING_ACTUAL;
        return comparePair(representative.relativePath, referencePath, actualPath, reportsRoot,
                options, missingStatus);
    }

    private BatchComparisonResult.FileResult comparePair(
            Path relativePath, Path referencePath, Path actualPath, Path reportsRoot,
            ImageComparator.ComparisonOptions options, BatchComparisonResult.Status missingStatus) {
        Path diagnostic = diagnosticPath(reportsRoot, relativePath);
        try {
            ImageComparator.ComparisonResult comparison =
                    imageComparator.compare(referencePath, actualPath, diagnostic, options);
            BatchComparisonResult.Status status = missingStatus != BatchComparisonResult.Status.DIFFERENT
                    ? missingStatus
                    : comparison.matches()
                    ? BatchComparisonResult.Status.MATCH
                    : BatchComparisonResult.Status.DIFFERENT;
            return fileResult(relativePath, status, referencePath, actualPath, comparison,
                    comparison.diagnosticImage(), comparison.compressedDiagnosticImage(), null);
        } catch (ImageComparator.ImageSizeMismatchException e) {
            return fileResult(relativePath, BatchComparisonResult.Status.SIZE_MISMATCH,
                    referencePath, actualPath, null, null, null, e.getMessage());
        } catch (IOException | IllegalArgumentException e) {
            if (missingStatus != BatchComparisonResult.Status.DIFFERENT) {
                return fileResult(relativePath, missingStatus, referencePath, actualPath,
                        null, null, null, e.getMessage());
            }
            return fileResult(relativePath, BatchComparisonResult.Status.ERROR,
                    referencePath, actualPath, null, null, null, e.getMessage());
        }
    }

    private static BatchComparisonResult.FileResult fileResult(
            Path relativePath, BatchComparisonResult.Status status,
            Path referencePath, Path actualPath,
            ImageComparator.ComparisonResult comparison,
            Path diagnostic, Path compressed, String message) {
        return new BatchComparisonResult.FileResult(relativePath, status, referencePath, actualPath,
                comparison, diagnostic, compressed, message);
    }

    private static Map<String, List<ImageFile>> discover(Path root) throws IOException {
        Map<String, List<ImageFile>> files = new TreeMap<>();
        try (Stream<Path> paths = Files.walk(root)) {
            List<Path> imagePaths = paths
                    .filter(path -> !Files.isSymbolicLink(path))
                    .filter(Files::isRegularFile)
                    .filter(path -> IMAGE_EXTENSIONS.contains(extension(path)))
                    .sorted(Comparator.comparing((Path path) -> normalizedRelative(root.relativize(path)))
                            .thenComparing(path -> portablePath(root.relativize(path))))
                    .toList();
            for (Path path : imagePaths) {
                Path relativePath = root.relativize(path);
                String extension = extension(path);
                String key = normalizedRelative(withoutExtension(relativePath));
                files.computeIfAbsent(key, ignored -> new ArrayList<>())
                        .add(new ImageFile(path, relativePath, extension));
            }
        }
        return files;
    }

    private static Path requireDirectory(Path path, String name) throws IOException {
        Path absolute = path.toAbsolutePath().normalize();
        if (!Files.isDirectory(absolute)) {
            throw new IOException(name + " is not a readable directory: " + absolute);
        }
        return absolute.toRealPath();
    }

    private static Path resolveIncludingMissingPath(Path absolute) throws IOException {
        ArrayDeque<Path> missingSegments = new ArrayDeque<>();
        Path existing = absolute;
        while (existing != null && !Files.exists(existing)) {
            missingSegments.push(existing.getFileName());
            existing = existing.getParent();
        }
        if (existing == null) {
            return absolute;
        }
        Path resolved = existing.toRealPath();
        while (!missingSegments.isEmpty()) {
            resolved = resolved.resolve(missingSegments.pop());
        }
        return resolved.normalize();
    }

    private static Path withoutExtension(Path relativePath) {
        Path filename = relativePath.getFileName();
        String value = filename.toString();
        int dot = value.lastIndexOf('.');
        String stem = dot > 0 ? value.substring(0, dot) : value;
        Path parent = relativePath.getParent();
        return parent == null ? Paths.get(stem) : parent.resolve(stem);
    }

    private static Path diagnosticPath(Path reportsRoot, Path relativePath) {
        Path filename = relativePath.getFileName();
        String value = filename.toString();
        int dot = value.lastIndexOf('.');
        String stem = dot > 0 ? value.substring(0, dot) : value;
        Path reportName = Paths.get(stem + "-diff.png");
        Path parent = relativePath.getParent();
        return reportsRoot.resolve(parent == null ? reportName : parent.resolve(reportName));
    }

    private static String extension(Path path) {
        String filename = path.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        return dot <= 0 || dot == filename.length() - 1
                ? "" : filename.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    private static String normalizedRelative(Path path) {
        return portablePath(path).toLowerCase(Locale.ROOT);
    }

    private static String portablePath(Path path) {
        return path.toString().replace('\\', '/');
    }

    private static String toJson(BatchComparisonResult result, Path reportsRoot) {
        StringBuilder json = new StringBuilder(1024);
        json.append('{').append('\n');
        appendCount(json, "totalFiles", result.totalFiles(), true);
        appendCount(json, "matched", result.matchedCount(), true);
        appendCount(json, "different", result.differentCount(), true);
        appendCount(json, "missingReference", result.missingReferenceCount(), true);
        appendCount(json, "missingActual", result.missingActualCount(), true);
        appendCount(json, "extensionMismatch", result.extensionMismatchCount(), true);
        appendCount(json, "sizeMismatch", result.sizeMismatchCount(), true);
        appendCount(json, "ambiguous", result.ambiguousCount(), true);
        appendCount(json, "errors", result.errorCount(), true);
        json.append("  \"files\": [");
        if (!result.files().isEmpty()) json.append('\n');
        for (int i = 0; i < result.files().size(); i++) {
            BatchComparisonResult.FileResult file = result.files().get(i);
            json.append("    {");
            json.append("\"relativePath\": ").append(quote(portablePath(file.relativePath())));
            json.append(", \"status\": ").append(quote(file.status().name()));
            if (file.comparison() != null) {
                ImageComparator.ComparisonResult comparison = file.comparison();
                json.append(", \"differentPixels\": ").append(comparison.differentPixels());
                json.append(", \"totalPixels\": ").append(comparison.totalPixels());
                json.append(", \"differencePercent\": ").append(comparison.differencePercent());
            }
            if (file.diagnosticImage() != null) {
                json.append(", \"diagnosticImage\": ")
                        .append(quote(portablePath(reportsRoot.relativize(file.diagnosticImage()))));
            }
            if (file.compressedDiagnosticImage() != null) {
                json.append(", \"compressedDiagnosticImage\": ")
                        .append(quote(portablePath(reportsRoot.relativize(file.compressedDiagnosticImage()))));
            }
            if (file.message() != null) {
                json.append(", \"message\": ").append(quote(file.message()));
            }
            json.append('}');
            if (i + 1 < result.files().size()) json.append(',');
            json.append('\n');
        }
        json.append("  ]\n");
        json.append('}');
        return json.toString();
    }

    private static void appendCount(StringBuilder json, String key, int value, boolean comma) {
        json.append("  ").append(quote(key)).append(": ").append(value);
        if (comma) json.append(',');
        json.append('\n');
    }

    private static String quote(String value) {
        StringBuilder encoded = new StringBuilder(value.length() + 2).append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> encoded.append("\\\"");
                case '\\' -> encoded.append("\\\\");
                case '\b' -> encoded.append("\\b");
                case '\f' -> encoded.append("\\f");
                case '\n' -> encoded.append("\\n");
                case '\r' -> encoded.append("\\r");
                case '\t' -> encoded.append("\\t");
                default -> {
                    if (c < 0x20) {
                        encoded.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        encoded.append(c);
                    }
                }
            }
        }
        return encoded.append('"').toString();
    }

    private static final class ImageFile {
        private final Path path;
        private final Path relativePath;
        private final String extension;

        private ImageFile(Path path, Path relativePath, String extension) {
            this.path = path;
            this.relativePath = relativePath;
            this.extension = extension;
        }
    }
}