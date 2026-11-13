package io.github.ahmedraafat2.imagecomparator;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable summary of a directory comparison.
 * Each file entry represents one relative image name across the two directories.
 */
public final class BatchComparisonResult {
    private final List<FileResult> files;
    private final Path summaryFile;
    private final Map<Status, Integer> counts;

    BatchComparisonResult(List<FileResult> files, Path summaryFile) {
        this.files = List.copyOf(Objects.requireNonNull(files, "files"));
        this.summaryFile = Objects.requireNonNull(summaryFile, "summaryFile");
        this.counts = new EnumMap<>(Status.class);
        for (Status status : Status.values()) {
            counts.put(status, 0);
        }
        for (FileResult file : this.files) {
            counts.compute(file.status(), (ignored, count) -> count + 1);
        }
    }

    /** Returns the number of unique relative image names found across both directories. */
    public int totalFiles() {
        return files.size();
    }

    public int matchedCount() {
        return count(Status.MATCH);
    }

    public int differentCount() {
        return count(Status.DIFFERENT);
    }

    public int missingReferenceCount() {
        return count(Status.MISSING_REFERENCE);
    }

    public int missingActualCount() {
        return count(Status.MISSING_ACTUAL);
    }

    public int extensionMismatchCount() {
        return count(Status.EXTENSION_MISMATCH);
    }

    public int sizeMismatchCount() {
        return count(Status.SIZE_MISMATCH);
    }

    public int ambiguousCount() {
        return count(Status.AMBIGUOUS);
    }

    public int errorCount() {
        return count(Status.ERROR);
    }

    /** Returns one entry per relative image name, in deterministic path order. */
    public List<FileResult> files() {
        return files;
    }

    /** Returns the generated JSON summary file. */
    public Path summaryFile() {
        return summaryFile;
    }

    private int count(Status status) {
        return counts.get(status);
    }

    public enum Status {
        MATCH,
        DIFFERENT,
        MISSING_REFERENCE,
        MISSING_ACTUAL,
        EXTENSION_MISMATCH,
        SIZE_MISMATCH,
        AMBIGUOUS,
        ERROR
    }

    /** Result for one relative image name. Nullable paths/results mean that step was not applicable. */
    public static final class FileResult {
        private final Path relativePath;
        private final Status status;
        private final Path referencePath;
        private final Path actualPath;
        private final ImageComparator.ComparisonResult comparison;
        private final Path diagnosticImage;
        private final Path compressedDiagnosticImage;
        private final String message;

        FileResult(Path relativePath, Status status, Path referencePath, Path actualPath,
                   ImageComparator.ComparisonResult comparison, Path diagnosticImage,
                   Path compressedDiagnosticImage, String message) {
            this.relativePath = Objects.requireNonNull(relativePath, "relativePath");
            this.status = Objects.requireNonNull(status, "status");
            this.referencePath = referencePath;
            this.actualPath = actualPath;
            this.comparison = comparison;
            this.diagnosticImage = diagnosticImage;
            this.compressedDiagnosticImage = compressedDiagnosticImage;
            this.message = message;
        }

        public Path relativePath() {
            return relativePath;
        }

        public Status status() {
            return status;
        }

        /** Returns the expected reference path, whether or not that file exists. */
        public Path referencePath() {
            return referencePath;
        }

        /** Returns the expected actual path, whether or not that file exists. */
        public Path actualPath() {
            return actualPath;
        }

        /** Returns pixel comparison metrics, or null if no pixel comparison was possible. */
        public ImageComparator.ComparisonResult comparison() {
            return comparison;
        }

        public Path diagnosticImage() {
            return diagnosticImage;
        }

        public Path compressedDiagnosticImage() {
            return compressedDiagnosticImage;
        }

        /** Returns an English explanation for extension conflicts, ambiguity, or errors; otherwise null. */
        public String message() {
            return message;
        }
    }
}