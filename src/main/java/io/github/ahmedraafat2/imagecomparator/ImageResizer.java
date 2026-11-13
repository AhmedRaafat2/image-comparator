package io.github.ahmedraafat2.imagecomparator;

import nu.pattern.OpenCV;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

/**
 * Resizes color images while preserving their aspect ratio.
 * Path-based resizing decodes images as BGR and does not preserve alpha transparency.
 * Uses INTER_AREA for shrinking and INTER_CUBIC for enlarging.
 */
public final class ImageResizer {
    static { OpenCV.loadLocally(); }

    private ImageResizer() { }

    /**
     * Resizes an image to {@code targetWidth}, preserving its aspect ratio, and writes it to the output path.
     * The output format is selected from the output file extension by OpenCV. Path-based input is decoded as BGR,
     * so alpha transparency is not preserved.
     *
     * @param inputPath path of the image to resize
     * @param outputPath destination path; may refer to the input to replace it in place
     * @param targetWidth desired output width in pixels; must be positive
     * @throws IOException if the input cannot be read or the output cannot be written
     * @throws IllegalArgumentException if targetWidth is not positive or the calculated height is unsupported
     * @throws NullPointerException if either path is null
     */
    public static void resizeToWidth(Path inputPath, Path outputPath, int targetWidth) throws IOException {
        resizeToWidth(inputPath, outputPath, targetWidth, 85);
    }

    /**
     * Resizes an image and, for JPEG output, uses the requested JPEG quality.
     * Quality is in the range 0 (smallest, lowest quality) to 100 (largest, highest quality).
     *
     * @param inputPath path of the image to resize
     * @param outputPath destination path
     * @param targetWidth desired output width in pixels; must be positive
     * @param jpegQuality JPEG quality; must be in the range 0..100
     * @throws IOException if the input cannot be read or the output cannot be written
     * @throws IllegalArgumentException if an argument or calculated dimension is unsupported
     */
    public static void resizeToWidth(Path inputPath, Path outputPath, int targetWidth, int jpegQuality)
            throws IOException {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        validateDimensions(targetWidth, jpegQuality);

        Mat source = null;
        try {
            source = Imgcodecs.imread(inputPath.toString(), Imgcodecs.IMREAD_COLOR);
            if (source.empty()) {
                throw new IOException("Unable to read image: " + inputPath);
            }
            writeResized(source, outputPath, targetWidth, jpegQuality);
        } catch (CvException e) {
            throw new IOException("OpenCV failed to resize image " + inputPath + " to " + outputPath, e);
        } finally {
            if (source != null) {
                source.release();
            }
        }
    }

    /**
     * Resizes an in-memory image without taking ownership of or modifying {@code source}.
     * This overload is package-private for reuse by the comparison report generator.
     */
    static void resizeToWidth(Mat source, Path outputPath, int targetWidth, int jpegQuality) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(outputPath, "outputPath");
        validateDimensions(targetWidth, jpegQuality);
        if (source.empty()) {
            throw new IOException("Unable to resize an empty image.");
        }
        try {
            writeResized(source, outputPath, targetWidth, jpegQuality);
        } catch (CvException e) {
            throw new IOException("OpenCV failed to resize image to " + outputPath, e);
        }
    }

    private static void validateDimensions(int targetWidth, int jpegQuality) {
        if (targetWidth <= 0) {
            throw new IllegalArgumentException("targetWidth must be positive");
        }
        if (jpegQuality < 0 || jpegQuality > 100) {
            throw new IllegalArgumentException("jpegQuality must be between 0 and 100");
        }
    }

    private static void writeResized(Mat source, Path outputPath, int targetWidth, int jpegQuality)
            throws IOException {
        long calculatedHeight = Math.round(targetWidth * (double) source.rows() / source.cols());
        if (calculatedHeight > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Calculated image height exceeds supported dimensions");
        }
        int targetHeight = (int) Math.max(1L, calculatedHeight);
        int interpolation = targetWidth < source.cols() ? Imgproc.INTER_AREA : Imgproc.INTER_CUBIC;

        Mat resized = null;
        MatOfInt encodingParams = null;
        try {
            resized = new Mat();
            Imgproc.resize(source, resized, new Size(targetWidth, targetHeight), 0, 0, interpolation);

            Path absoluteOutput = outputPath.toAbsolutePath();
            Path parent = absoluteOutput.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String outputName = absoluteOutput.toString().toLowerCase(Locale.ROOT);
            boolean jpeg = outputName.endsWith(".jpg") || outputName.endsWith(".jpeg");
            boolean saved;
            if (jpeg) {
                encodingParams = new MatOfInt();
                encodingParams.fromArray(new int[] {Imgcodecs.IMWRITE_JPEG_QUALITY, jpegQuality});
                saved = Imgcodecs.imwrite(absoluteOutput.toString(), resized, encodingParams);
            } else {
                saved = Imgcodecs.imwrite(absoluteOutput.toString(), resized);
            }
            if (!saved) {
                throw new IOException("OpenCV could not write resized image: " + absoluteOutput);
            }
        } finally {
            if (resized != null) {
                resized.release();
            }
            if (encodingParams != null) {
                encodingParams.release();
            }
        }
    }
}
