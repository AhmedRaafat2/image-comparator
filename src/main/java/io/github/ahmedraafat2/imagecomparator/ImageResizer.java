package io.github.ahmedraafat2.imagecomparator;

import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Resizes an image to a target width while preserving its aspect ratio.
 * OpenCV chooses an interpolation method suited to shrinking or enlarging.
 */
public final class ImageResizer {
    static { OpenCV.loadLocally(); }

    private ImageResizer() { }

    /**
     * Resizes an image to {@code targetWidth}, preserving its aspect ratio, and writes it to the output path.
     * The output format is selected from the output file extension by OpenCV.
     *
     * @param inputPath path of the image to resize
     * @param outputPath destination path
     * @param targetWidth desired output width in pixels; must be positive
     * @throws IOException if the input cannot be read or the output cannot be written
     * @throws IllegalArgumentException if targetWidth is not positive
     * @throws NullPointerException if either path is null
     */
    public static void resizeToWidth(Path inputPath, Path outputPath, int targetWidth) throws IOException {
        resizeToWidth(inputPath, outputPath, targetWidth, 85);
    }

    /**
     * Resizes an image to a target width and, for JPEG output, uses the requested JPEG quality.
     * Quality is in the range 0 (smallest, lowest quality) to 100 (largest, highest quality).
     */
    public static void resizeToWidth(Path inputPath, Path outputPath, int targetWidth, int jpegQuality)
            throws IOException {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        if (targetWidth <= 0) {
            throw new IllegalArgumentException("targetWidth must be positive");
        }
        if (jpegQuality < 0 || jpegQuality > 100) {
            throw new IllegalArgumentException("jpegQuality must be between 0 and 100");
        }

        Mat image = Imgcodecs.imread(inputPath.toString(), Imgcodecs.IMREAD_COLOR);
        Mat resized = new Mat();
        MatOfInt encodingParams = new MatOfInt();
        try {
            if (image.empty()) {
                throw new IOException("Unable to read image: " + inputPath);
            }

            int targetHeight = Math.max(1, (int) Math.round(targetWidth * (double) image.rows() / image.cols()));
            int interpolation = targetWidth < image.cols() ? Imgproc.INTER_AREA : Imgproc.INTER_CUBIC;
            Imgproc.resize(image, resized, new Size(targetWidth, targetHeight), 0, 0, interpolation);

            Path absoluteOutput = outputPath.toAbsolutePath();
            Path parent = absoluteOutput.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String outputName = absoluteOutput.toString().toLowerCase(java.util.Locale.ROOT);
            boolean jpeg = outputName.endsWith(".jpg") || outputName.endsWith(".jpeg");
            boolean saved;
            if (jpeg) {
                encodingParams.fromArray(new int[] {Imgcodecs.IMWRITE_JPEG_QUALITY, jpegQuality});
                saved = Imgcodecs.imwrite(absoluteOutput.toString(), resized, encodingParams);
            } else {
                saved = Imgcodecs.imwrite(absoluteOutput.toString(), resized);
            }
            if (!saved) {
                throw new IOException("OpenCV could not write resized image: " + absoluteOutput);
            }
        } finally {
            image.release();
            resized.release();
            encodingParams.release();
        }
    }
}
