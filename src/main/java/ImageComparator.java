import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * A utility class for comparing two images pixel by pixel using OpenCV.
 * <p>
 * The class provides a method to:
 * <ul>
 *   <li>Load and preprocess reference and test images</li>
 *   <li>Compare images using absolute pixel-wise difference</li>
 *   <li>Highlight differences with red circles</li>
 *   <li>Save a composite image showing the reference, test, and diff results</li>
 * </ul>
 * <p>
 * This can be used for visual regression testing, automated image validations, etc.
 */
public class ImageComparator {

    // Load the native OpenCV library
    static {
        OpenCV.loadLocally();
    }

    /**
     * Compares two images pixel by pixel and highlights any visual differences using red circles.
     * A composite image is saved if differences are detected.
     *
     * @param referenceImagePath Path to the reference image
     * @param currentImagePath   Path to the current (actual) image
     * @param differenceImagePath      Path to save the result image (reference + actual + diff)
     * @param threshold          Matching threshold (e.g., 0.99 means 99% match required)
     * @return {@code true} if the images match within the threshold, otherwise {@code false}
     */
    public boolean compareAndSaveDiffIfNotMatch(String referenceImagePath, String currentImagePath, String differenceImagePath, double threshold) {

        // 🟢 Load images from file system
        Mat reference = Imgcodecs.imread(referenceImagePath);
        Mat current = Imgcodecs.imread(currentImagePath);

        // Validate image loading
        if (reference.empty() || current.empty()) {
            System.out.println("❌ Failed to load one or both images.");
            return false;
        }

        // 🟢 Resize images if dimensions differ
        if (!reference.size().equals(current.size())) {
            System.out.println("⚠️ Image sizes differ. Resizing to match...");

            Size targetSize = new Size(
                    Math.min(reference.width(), current.width()),
                    Math.min(reference.height(), current.height())
            );

            Imgproc.resize(reference, reference, targetSize);
            Imgproc.resize(current, current, targetSize);
        }

        // 🟢 Compute absolute difference between the two images
        Mat diff = new Mat();
        Core.absdiff(reference, current, diff);

        // Convert diff image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(diff, gray, Imgproc.COLOR_BGR2GRAY);

        // Apply binary threshold to highlight non-zero differences
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 1, 255, Imgproc.THRESH_BINARY);

        // Count non-zero (different) pixels
        int nonZero = Core.countNonZero(binary);
        double totalPixels = binary.rows() * binary.cols();
        double differencePercentage = (nonZero / totalPixels) * 100;

        System.out.printf("📊 Difference percentage: %.5f%% (%d pixels)%n", differencePercentage, nonZero);

        // Determine if images match within the allowed threshold
        boolean isMatch = differencePercentage <= (100 - threshold * 100);

        // 🟢 If not matched, prepare visual diff image
        if (!isMatch) {

            // Ensure output directory exists
            File diffFile = new File(differenceImagePath);
            File parentDir = diffFile.getParentFile();
            if (!parentDir.exists()) {
                boolean created = parentDir.mkdirs();
                if (created) {
                    System.out.println("📂 Created directory: " + parentDir.getAbsolutePath());
                } else {
                    System.out.println("❌ Failed to create directory: " + parentDir.getAbsolutePath());
                }
            }

            // 🟢 Extract contours (areas with differences)
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // 🟢 Draw red circles around the different regions on the current image
            for (MatOfPoint contour : contours) {
                Point center = new Point();
                float[] radius = new float[1];
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                Imgproc.minEnclosingCircle(contour2f, center, radius);
                Imgproc.circle(current, center, (int) radius[0], new Scalar(0, 0, 255), 2);
            }

            // 🟢 Create a side-by-side composite image (reference | current | diff)
            Mat combined = new Mat(reference.rows(), reference.cols() * 3, reference.type());
            reference.copyTo(combined.colRange(0, reference.cols()));
            current.copyTo(combined.colRange(reference.cols(), reference.cols() * 2));
            diff.copyTo(combined.colRange(reference.cols() * 2, reference.cols() * 3));

            // Save the composite diff image
            boolean saved = Imgcodecs.imwrite(differenceImagePath, combined);
            if (saved) {
                System.out.println("❌ Images differ - Diff image saved to: " + differenceImagePath);
            } else {
                System.out.println("❌ Images differ - Failed to save diff image.");
            }
        }

        return isMatch;
    }

    /**
     * Compares two images pixel by pixel and returns whether they match within the given threshold.
     * This method does not save any output images or visualize the differences.
     *
     * @param referenceImagePath Path to the reference image
     * @param currentImagePath   Path to the current (actual) image
     * @param threshold          Matching threshold (e.g., 0.99 means 99% match required)
     * @return {@code true} if the images match within the threshold, otherwise {@code false}
     */
    public boolean areImagesSimilar(String referenceImagePath, String currentImagePath, double threshold) {
        // 🟢 Load images
        Mat reference = Imgcodecs.imread(referenceImagePath);
        Mat current = Imgcodecs.imread(currentImagePath);

        // Validate image loading
        if (reference.empty() || current.empty()) {
            System.out.println("❌ Failed to load one or both images.");
            return false;
        }

        // 🟢 Resize if necessary to match dimensions
        if (!reference.size().equals(current.size())) {
            Size targetSize = new Size(
                    Math.min(reference.width(), current.width()),
                    Math.min(reference.height(), current.height())
            );

            Imgproc.resize(reference, reference, targetSize);
            Imgproc.resize(current, current, targetSize);
        }

        // 🟢 Compute absolute difference
        Mat diff = new Mat();
        Core.absdiff(reference, current, diff);

        // Convert diff to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(diff, gray, Imgproc.COLOR_BGR2GRAY);

        // Apply threshold to get binary image
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 1, 255, Imgproc.THRESH_BINARY);

        // Count differing pixels
        int nonZero = Core.countNonZero(binary);
        double totalPixels = binary.rows() * binary.cols();
        double differencePercentage = (nonZero / totalPixels) * 100;

        System.out.printf("📊 Difference percentage: %.5f%% (%d pixels)%n", differencePercentage, nonZero);

        // Return whether images are similar enough
        return differencePercentage <= (100 - threshold * 100);
    }

}
