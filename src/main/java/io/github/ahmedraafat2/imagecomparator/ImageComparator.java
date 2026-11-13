package io.github.ahmedraafat2.imagecomparator;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Pixel-oriented image comparison with optional three-panel diagnostic output.
 * Input images are decoded as BGR, so source alpha transparency is not preserved.
 * Supported input and output formats depend on the codecs bundled with OpenCV.
 */
public final class ImageComparator {
    static { OpenCV.loadLocally(); }

    /**
     * Compares two files with exact-pixel defaults and without creating an output file.
     * @param referencePath baseline image
     * @param actualPath image to compare
     * @return comparison metrics and match state
     * @throws IOException if diagnostic output was requested and cannot be written
     */
    public ComparisonResult compare(Path referencePath, Path actualPath) throws IOException {
        return compare(referencePath, actualPath, null, ComparisonOptions.defaults());
    }

    /**
     * Compares two files with custom thresholds and policies, without creating an output file.
     * @param referencePath baseline image
     * @param actualPath image to compare
     * @param options comparison thresholds and behavior
     * @return comparison metrics and match state
     * @throws IOException if diagnostic output was requested and cannot be written
     */
    public ComparisonResult compare(Path referencePath, Path actualPath, ComparisonOptions options) throws IOException {
        return compare(referencePath, actualPath, null, options);
    }

    /**
     * Compares images and writes a diagnostic triptych when they differ or an input is missing.
     * The panels are reference, actual image with red marks, and grayscale difference. A reduced JPEG copy is also written.
     * @param referencePath baseline image path
     * @param actualPath image path to compare
     * @param diagnosticOutput output path for the three-panel image; null disables output
     * @param options comparison thresholds and behavior
     * @return counts, dimensions, percentage, and match state
     * @throws IOException if an image cannot be read or the diagnostic cannot be written
     * @throws NullPointerException if a required path or options object is null
     * @throws IllegalArgumentException if extensions differ or dimensions differ under the FAIL size policy
     * @throws IllegalArgumentException if the diagnostic output would overwrite either input image
     */
    public ComparisonResult compare(Path referencePath, Path actualPath, Path diagnosticOutput,
                                    ComparisonOptions options) throws IOException {
        Objects.requireNonNull(referencePath, "referencePath");
        Objects.requireNonNull(actualPath, "actualPath");
        Objects.requireNonNull(options, "options");
        String referenceExtension = extension(referencePath);
        String actualExtension = extension(actualPath);
        if (!referenceExtension.equalsIgnoreCase(actualExtension)) {
            throw new IllegalArgumentException("Image extensions must match: reference=" +
                    referenceExtension + ", actual=" + actualExtension);
        }

        Mat reference = null;
        Mat actual = null;
        Mat diff = null;
        Mat mask = null;
        boolean missingImage = false;
        try {
            reference = Imgcodecs.imread(referencePath.toString(), Imgcodecs.IMREAD_COLOR);
            actual = Imgcodecs.imread(actualPath.toString(), Imgcodecs.IMREAD_COLOR);
            diff = new Mat();
            mask = new Mat();
            boolean referenceMissing = reference.empty();
            boolean actualMissing = actual.empty();
            missingImage = referenceMissing || actualMissing;
            if (referenceMissing && actualMissing) {
                reference.release();
                actual.release();
                reference = missingPlaceholder(800, 600);
                actual = missingPlaceholder(800, 600);
            } else if (referenceMissing) {
                reference.release();
                reference = missingPlaceholder(actual.cols(), actual.rows());
            } else if (actualMissing) {
                actual.release();
                actual = missingPlaceholder(reference.cols(), reference.rows());
            }

            if (!reference.size().equals(actual.size())) {
                if (options.sizeMismatchPolicy == SizeMismatchPolicy.FAIL) {
                    throw new IllegalArgumentException("Image dimensions differ: reference=" +
                            reference.cols() + "x" + reference.rows() + ", actual=" +
                            actual.cols() + "x" + actual.rows());
                }
                if (options.sizeMismatchPolicy == SizeMismatchPolicy.RESIZE_ACTUAL_TO_REFERENCE) {
                    Imgproc.resize(actual, actual, reference.size(), 0, 0, Imgproc.INTER_AREA);
                } else {
                    Size minimum = new Size(Math.min(reference.cols(), actual.cols()),
                            Math.min(reference.rows(), actual.rows()));
                    Imgproc.resize(reference, reference, minimum, 0, 0, Imgproc.INTER_AREA);
                    Imgproc.resize(actual, actual, minimum, 0, 0, Imgproc.INTER_AREA);
                }
            }

            Core.absdiff(reference, actual, diff);
            Core.inRange(diff, Scalar.all(0), Scalar.all(options.pixelThreshold), mask);
            Core.bitwise_not(mask, mask);
            int changed = Core.countNonZero(mask);
            long total = (long) mask.rows() * mask.cols();
            double percent = total == 0 ? 0 : changed * 100.0 / total;
            boolean matches = !missingImage && percent <= options.allowedDifferencePercent;
            Path compressedOutput = null;
            if ((!matches || missingImage) && diagnosticOutput != null) {
                validateDiagnosticPaths(referencePath, actualPath, diagnosticOutput);
                compressedOutput = writeDiagnostic(reference, actual, diff, mask, diagnosticOutput, options);
            }
            return new ComparisonResult(matches, reference.cols(), reference.rows(), changed, total, percent,
                    matches ? null : diagnosticOutput, compressedOutput);
        } catch (CvException e) {
            throw new IOException("OpenCV failed to compare images " + referencePath + " and " + actualPath, e);
        } finally {
            if (reference != null) reference.release();
            if (actual != null) actual.release();
            if (diff != null) diff.release();
            if (mask != null) mask.release();
        }
    }

    private static void validateDiagnosticPaths(Path reference, Path actual, Path diagnostic) throws IOException {
        Path absoluteDiagnostic = diagnostic.toAbsolutePath().normalize();
        Path compressed = compressedPath(absoluteDiagnostic).toAbsolutePath().normalize();
        if (sameFileOrPath(absoluteDiagnostic, reference) || sameFileOrPath(absoluteDiagnostic, actual)
                || sameFileOrPath(compressed, reference) || sameFileOrPath(compressed, actual)) {
            throw new IllegalArgumentException("Diagnostic output paths must not overwrite either input image");
        }
    }

    private static boolean sameFileOrPath(Path first, Path second) throws IOException {
        Path absoluteSecond = second.toAbsolutePath().normalize();
        if (first.equals(absoluteSecond)) return true;
        return Files.exists(first) && Files.exists(absoluteSecond) && Files.isSameFile(first, absoluteSecond);
    }

    private static String extension(Path path) {
        Path name = path.getFileName();
        String value = name == null ? "" : name.toString();
        int dot = value.lastIndexOf('.');
        if (dot <= 0 || dot == value.length() - 1) {
            throw new IllegalArgumentException("Image path must have a file extension: " + path);
        }
        return value.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    private static Mat missingPlaceholder(int width, int height) {
        Mat placeholder = new Mat(height, width, CvType.CV_8UC3, Scalar.all(255));
        String label = "Image not found";
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;
        double scale = Math.min(1.2, Math.max(0.35, width / 500.0));
        int thickness = Math.max(1, (int) Math.round(scale * 2));
        int[] baseline = new int[1];
        Size textSize = Imgproc.getTextSize(label, font, scale, thickness, baseline);
        Point origin = new Point(Math.max(2, (width - textSize.width) / 2),
                Math.max((int) textSize.height + 2, (height + textSize.height) / 2));
        Imgproc.putText(placeholder, label, origin, font, scale, new Scalar(0, 0, 0), thickness, Imgproc.LINE_AA);
        return placeholder;
    }
    private static Path writeDiagnostic(Mat reference, Mat actual, Mat diff, Mat mask, Path output,
                                        ComparisonOptions options) throws IOException {
        Path absolute = output.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent != null) Files.createDirectories(parent);

        Mat marked = null;
        Mat displayDiff = null;
        Mat composite = null;
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = null;
        Mat groupingKernel = null;
        Mat contourInput = null;
        try {
            marked = actual.clone();
            displayDiff = new Mat();
            composite = new Mat();
            hierarchy = new Mat();
            groupingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
            contourInput = new Mat();
            Imgproc.dilate(mask, contourInput, groupingKernel);
            Imgproc.findContours(contourInput, contours, hierarchy, Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                if (Imgproc.contourArea(contour) < options.minimumRegionArea) continue;
                MatOfPoint2f points = new MatOfPoint2f(contour.toArray());
                try {
                    Point center = new Point();
                    float[] radius = new float[1];
                    Imgproc.minEnclosingCircle(points, center, radius);
                    Imgproc.circle(marked, center, Math.max(1, (int) Math.ceil(radius[0])),
                            new Scalar(0, 0, 255), options.circleThickness, Imgproc.LINE_AA);
                } finally {
                    points.release();
                }
            }
            Imgproc.cvtColor(diff, displayDiff, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(displayDiff, displayDiff, Imgproc.COLOR_GRAY2BGR);
            Core.hconcat(java.util.Arrays.asList(reference, marked, displayDiff), composite);
            if (!Imgcodecs.imwrite(absolute.toString(), composite)) {
                throw new IOException("OpenCV could not write diagnostic image: " + absolute);
            }
            Path compressed = compressedPath(absolute);
            int compressedWidth = Math.min(composite.cols(), 1200);
            ImageResizer.resizeToWidth(composite, compressed, compressedWidth, 80);
            return compressed;
        } catch (CvException e) {
            throw new IOException("OpenCV failed to create diagnostic image: " + absolute, e);
        } finally {
            if (marked != null) marked.release();
            if (displayDiff != null) displayDiff.release();
            if (composite != null) composite.release();
            if (hierarchy != null) hierarchy.release();
            if (groupingKernel != null) groupingKernel.release();
            if (contourInput != null) contourInput.release();
            for (MatOfPoint contour : contours) contour.release();
        }
    }

    private static Path compressedPath(Path output) {
        String filename = output.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        String base = dot <= 0 ? filename : filename.substring(0, dot);
        return output.resolveSibling(base + "-compressed.jpg");
    }

    /** A pixel is changed when any BGR channel differs by more than pixelThreshold. */
    public static final class ComparisonOptions {
        private final double allowedDifferencePercent;
        private final double pixelThreshold;
        private final double minimumRegionArea;
        private final int circleThickness;
        private final SizeMismatchPolicy sizeMismatchPolicy;

        private ComparisonOptions(Builder b) {
            allowedDifferencePercent = b.allowedDifferencePercent;
            pixelThreshold = b.pixelThreshold;
            minimumRegionArea = b.minimumRegionArea;
            circleThickness = b.circleThickness;
            sizeMismatchPolicy = b.sizeMismatchPolicy;
        }
        public static ComparisonOptions defaults() { return builder().build(); }
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private double allowedDifferencePercent = 0.0;
            private double pixelThreshold = 0.0;
            private double minimumRegionArea = 0.0;
            private int circleThickness = 2;
            private SizeMismatchPolicy sizeMismatchPolicy = SizeMismatchPolicy.FAIL;
            public Builder allowedDifferencePercent(double value) {
                if (!Double.isFinite(value) || value < 0 || value > 100) throw new IllegalArgumentException("allowedDifferencePercent must be 0..100");
                allowedDifferencePercent = value; return this;
            }
            public Builder pixelThreshold(double value) {
                if (!Double.isFinite(value) || value < 0 || value > 255) throw new IllegalArgumentException("pixelThreshold must be 0..255");
                pixelThreshold = value; return this;
            }
            public Builder minimumRegionArea(double value) {
                if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("minimumRegionArea must be >= 0");
                minimumRegionArea = value; return this;
            }
            public Builder circleThickness(int value) {
                if (value < 1) throw new IllegalArgumentException("circleThickness must be >= 1");
                circleThickness = value; return this;
            }
            public Builder sizeMismatchPolicy(SizeMismatchPolicy value) {
                sizeMismatchPolicy = Objects.requireNonNull(value); return this;
            }
            public ComparisonOptions build() { return new ComparisonOptions(this); }
        }
    }

    public enum SizeMismatchPolicy { FAIL, RESIZE_ACTUAL_TO_REFERENCE, RESIZE_BOTH_TO_MINIMUM }

    public static final class ComparisonResult {
        private final boolean matches;
        private final int width, height;
        private final int differentPixels;
        private final long totalPixels;
        private final double differencePercent;
        private final Path diagnosticImage;
        private final Path compressedDiagnosticImage;
        private ComparisonResult(boolean matches, int width, int height, int differentPixels, long totalPixels,
                                 double differencePercent, Path diagnosticImage, Path compressedDiagnosticImage) {
            this.matches = matches; this.width = width; this.height = height;
            this.differentPixels = differentPixels; this.totalPixels = totalPixels;
            this.differencePercent = differencePercent; this.diagnosticImage = diagnosticImage;
            this.compressedDiagnosticImage = compressedDiagnosticImage;
        }
        public boolean matches() { return matches; }
        public int width() { return width; }
        public int height() { return height; }
        public int differentPixels() { return differentPixels; }
        public long totalPixels() { return totalPixels; }
        public double differencePercent() { return differencePercent; }
        public Path diagnosticImage() { return diagnosticImage; }
        /** Returns the reduced-size JPEG diagnostic, or null when none was created. */
        public Path compressedDiagnosticImage() { return compressedDiagnosticImage; }
        @Override public String toString() {
            return "ComparisonResult{matches=" + matches + ", differentPixels=" + differentPixels +
                    "/" + totalPixels + ", differencePercent=" + differencePercent + "}";
        }
    }
}
