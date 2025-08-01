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
     * Compares two images and saves a difference image if the mismatch exceeds a given threshold.
     * <p>
     * If one of the images is missing (either reference or current), it will be replaced with a white image of the same size
     * (based on the existing image). If both are missing, two white images of default size (1920x1080) will be created.
     * </p>
     *
     * @param referenceImagePath           Path to the reference (baseline) image.
     * @param currentImagePath             Path to the current image to be compared.
     * @param differenceImagePath          Path to save the visual diff image if images don't match.
     * @param differenceThresholdPercentage The allowed percentage of pixel difference (e.g., 5.0 means 5% is allowed).
     * @return true if the images are considered a match (i.e., difference is within threshold); false otherwise.
     */
    public boolean compareAndSaveDiffIfNotMatch(
            String referenceImagePath,
            String currentImagePath,
            String differenceImagePath,
            double differenceThresholdPercentage
    ) {
        // Load images from file
        Mat reference = Imgcodecs.imread(referenceImagePath);
        Mat current = Imgcodecs.imread(currentImagePath);

        boolean refMissing = reference.empty();
        boolean curMissing = current.empty();

        // Handle missing images
        if (refMissing && curMissing) {
            System.out.println("⚠️ كلا الصورتين غير موجودتين. سيتم إنشاء صور بيضاء للمقارنة.");
            reference = Mat.ones(1920, 1080, CvType.CV_8UC3);
            reference.setTo(new Scalar(255, 255, 255));

            current = Mat.ones(1920, 1080, CvType.CV_8UC3);
            current.setTo(new Scalar(255, 255, 255));
        } else if (refMissing) {
            System.out.println("⚠️ صورة المرجع غير موجودة. سيتم إنشاء صورة بيضاء بنفس حجم الحالية.");
            reference = Mat.ones(current.rows(), current.cols(), current.type());
            reference.setTo(new Scalar(255, 255, 255));
        } else if (curMissing) {
            System.out.println("⚠️ الصورة الحالية غير موجودة. سيتم إنشاء صورة بيضاء بنفس حجم المرجع.");
            current = Mat.ones(reference.rows(), reference.cols(), reference.type());
            current.setTo(new Scalar(255, 255, 255));
        }

        // Resize if sizes don't match
        if (!reference.size().equals(current.size())) {
            System.out.println("⚠️ Image sizes differ. Resizing 'current' image to match 'reference'...");
            Imgproc.resize(current, current, reference.size());
        }

        // Compute absolute difference between the two images
        Mat diff = new Mat();
        Core.absdiff(reference, current, diff);

        // Convert diff image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(diff, gray, Imgproc.COLOR_BGR2GRAY);

        // Threshold the grayscale image to highlight different pixels
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 1.0, 255.0, Imgproc.THRESH_BINARY);

        // Count non-zero (different) pixels
        int nonZeroPixels = Core.countNonZero(binary);
        double totalPixels = (double) binary.rows() * binary.cols();
        double differencePercentage = (nonZeroPixels / totalPixels) * 100.0;

        // Print statistics
        System.out.printf("📊 Difference percentage: %.5f%% (%d different pixels)%n", differencePercentage, nonZeroPixels);

        // Determine if the images are within acceptable difference
        boolean isMatch = differencePercentage <= differenceThresholdPercentage;

        if (!isMatch) {
            System.out.println("❌ Images do not match. Generating difference image...");

            // Ensure parent directory exists
            File diffFile = new File(differenceImagePath);
            File parentDir = diffFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                if (parentDir.mkdirs()) {
                    System.out.println("📂 Created directory: " + parentDir.getAbsolutePath());
                } else {
                    System.out.println("❌ Failed to create directory: " + parentDir.getAbsolutePath());
                }
            }

            // Detect and draw contours (highlight differences with circles)
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (MatOfPoint contour : contours) {
                Point center = new Point();
                float[] radius = new float[1];
                Imgproc.minEnclosingCircle(new MatOfPoint2f(contour.toArray()), center, radius);
                Imgproc.circle(current, center, (int) radius[0], new Scalar(0, 0, 255), 2); // Red circle
                contour.release();
            }
            hierarchy.release();

            // Create a combined image showing: reference | current | diff
            Mat combined = new Mat(reference.rows(), reference.cols() * 3, reference.type());
            reference.copyTo(combined.colRange(0, reference.cols()));
            current.copyTo(combined.colRange(reference.cols(), reference.cols() * 2));
            diff.copyTo(combined.colRange(reference.cols() * 2, reference.cols() * 3));

            // Save the combined image
            if (Imgcodecs.imwrite(differenceImagePath, combined)) {
                System.out.println("✅ Diff image saved to: " + differenceImagePath);
            } else {
                System.out.println("❌ Failed to save diff image.");
            }

            combined.release();
        } else {
            System.out.println("✅ Images match within the given threshold.");
        }

        // Clean up memory
        reference.release();
        current.release();
        diff.release();
        gray.release();
        binary.release();

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
