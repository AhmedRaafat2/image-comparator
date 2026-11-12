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
import java.util.Objects;

/**
 * Pixel-oriented image comparison with optional three-panel diagnostic output.
 * Supported input and output formats depend on the codecs bundled with OpenCV.
 */
public final class ImageComparator {
    static { OpenCV.loadLocally(); }

    /**
     * Compares two files with exact-pixel defaults and without creating an output file.
     * @param referencePath baseline image
     * @param actualPath image to compare
     * @return comparison metrics and match state
     * @throws IOException if either image cannot be read
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
     * @throws IOException if either image cannot be read
     */
    public ComparisonResult compare(Path referencePath, Path actualPath, ComparisonOptions options) throws IOException {
        return compare(referencePath, actualPath, null, options);
    }

    /**
     * Compares images and writes a diagnostic triptych when the allowed difference is exceeded.
     * The panels are reference, actual image with red circles, and grayscale difference.
     * @param referencePath baseline image path
     * @param actualPath image path to compare
     * @param diagnosticOutput output path for the three-panel image; null disables output
     * @param options comparison thresholds and behavior
     * @return counts, dimensions, percentage, and match state
     * @throws IOException if an image cannot be read or the diagnostic cannot be written
     */
    public ComparisonResult compare(Path referencePath, Path actualPath, Path diagnosticOutput,
                                    ComparisonOptions options) throws IOException {
        Objects.requireNonNull(referencePath, "referencePath");
        Objects.requireNonNull(actualPath, "actualPath");
        Objects.requireNonNull(options, "options");
        Mat reference = Imgcodecs.imread(referencePath.toString(), Imgcodecs.IMREAD_COLOR);
        Mat actual = Imgcodecs.imread(actualPath.toString(), Imgcodecs.IMREAD_COLOR);
        Mat diff = new Mat(), mask = new Mat();
        try {
            if (reference.empty()) throw new IOException("Unable to read reference image: " + referencePath);
            if (actual.empty()) throw new IOException("Unable to read actual image: " + actualPath);
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
            boolean matches = percent <= options.allowedDifferencePercent;
            if (!matches && diagnosticOutput != null) {
                writeDiagnostic(reference, actual, diff, mask, diagnosticOutput, options);
            }
            return new ComparisonResult(matches, reference.cols(), reference.rows(), changed, total, percent,
                    matches ? null : diagnosticOutput);
        } finally {
            reference.release();
            actual.release();
            diff.release();
            mask.release();
        }
    }

    private static void writeDiagnostic(Mat reference, Mat actual, Mat diff, Mat mask, Path output,
                                        ComparisonOptions options) throws IOException {
        Path absolute = output.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent != null) Files.createDirectories(parent);

        Mat marked = actual.clone();
        Mat displayDiff = new Mat();
        Mat composite = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat contourInput = mask.clone();
        try {
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
        } finally {
            marked.release();
            displayDiff.release();
            composite.release();
            hierarchy.release();
            contourInput.release();
            for (MatOfPoint contour : contours) contour.release();
        }
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
        private ComparisonResult(boolean matches, int width, int height, int differentPixels, long totalPixels,
                                 double differencePercent, Path diagnosticImage) {
            this.matches = matches; this.width = width; this.height = height;
            this.differentPixels = differentPixels; this.totalPixels = totalPixels;
            this.differencePercent = differencePercent; this.diagnosticImage = diagnosticImage;
        }
        public boolean matches() { return matches; }
        public int width() { return width; }
        public int height() { return height; }
        public int differentPixels() { return differentPixels; }
        public long totalPixels() { return totalPixels; }
        public double differencePercent() { return differencePercent; }
        public Path diagnosticImage() { return diagnosticImage; }
        @Override public String toString() {
            return "ComparisonResult{matches=" + matches + ", differentPixels=" + differentPixels +
                    "/" + totalPixels + ", differencePercent=" + differencePercent + "}";
        }
    }
}
