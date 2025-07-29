import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for comparing images using OpenCV.
 * Provides pixel-by-pixel comparison with difference visualization.
 *
 * <p><b>Comparison Algorithm:</b></p>
 * <ol>
 *   <li>Load reference and current images</li>
 *   <li>Convert to grayscale</li>
 *   <li>Calculate absolute difference</li>
 *   <li>Apply binary threshold</li>
 *   <li>Calculate difference percentage</li>
 *   <li>Generate composite image if differences found</li>
 * </ol>
 */
public class OpenCVComparator {

    static {
        OpenCV.loadLocally();
    }

    /**
     * يقارن بين صورتين بيكسل بيكسل ويحدد الاختلاف بدوائر.
     *
     * @param referenceImagePath المسار للصورة المرجعية
     * @param currentImagePath المسار للصورة الحالية
     * @param diffImagePath المسار لحفظ صورة المقارنة
     * @param threshold نسبة التشابه المقبولة (مثلا 0.99)
     * @return true لو الصور متطابقة
     */
    public boolean compareImagesPixelByPixel(String referenceImagePath, String currentImagePath, String diffImagePath, double threshold) {
        // 🟢 تحميل الصور
        Mat reference = Imgcodecs.imread(referenceImagePath);
        Mat current = Imgcodecs.imread(currentImagePath);

        if (reference.empty() || current.empty()) {
            System.out.println("❌ لم أتمكن من تحميل الصور.");
            return false;
        }


        if (!reference.size().equals(current.size())) {
            System.out.println("⚠️ حجم الصور مختلف. جاري إعادة التحجيم...");

            Size targetSize = new Size(
                    Math.min(reference.width(), current.width()),
                    Math.min(reference.height(), current.height())
            );

            Imgproc.resize(reference, reference, targetSize);
            Imgproc.resize(current, current, targetSize);
        }

        // 🟢 حساب الفرق
        Mat diff = new Mat();
        Core.absdiff(reference, current, diff);

        Mat gray = new Mat();
        Imgproc.cvtColor(diff, gray, Imgproc.COLOR_BGR2GRAY);

        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 1, 255, Imgproc.THRESH_BINARY);

        int nonZero = Core.countNonZero(binary);
        double totalPixels = binary.rows() * binary.cols();
        double differencePercentage = (nonZero / totalPixels) * 100;

        System.out.printf("📊 نسبة البكسلات المختلفة: %.5f%% (%d بكسل)%n", differencePercentage, nonZero);

        boolean isMatch = differencePercentage <= (100 - threshold * 100);

        if (!isMatch) {
            // 🟢 تجهيز المسار
            File diffFile = new File(diffImagePath);
            File parentDir = diffFile.getParentFile();
            if (!parentDir.exists()) {
                boolean created = parentDir.mkdirs();
                if (created) {
                    System.out.println("📂 تم إنشاء المجلد: " + parentDir.getAbsolutePath());
                } else {
                    System.out.println("❌ فشل إنشاء المجلد: " + parentDir.getAbsolutePath());
                }
            }

            // 🟢 استخراج الكونتورز
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // 🟢 رسم دوائر حمراء حول الاختلافات
            for (MatOfPoint contour : contours) {
                Point center = new Point();
                float[] radius = new float[1];
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                Imgproc.minEnclosingCircle(contour2f, center, radius);
                Imgproc.circle(current, center, (int) radius[0], new Scalar(0, 0, 255), 2);
            }

            // 🟢 إنشاء الصورة المركبة
            Mat combined = new Mat(reference.rows(), reference.cols() * 3, reference.type());
            reference.copyTo(combined.colRange(0, reference.cols()));
            current.copyTo(combined.colRange(reference.cols(), reference.cols() * 2));
            diff.copyTo(combined.colRange(reference.cols() * 2, reference.cols() * 3));

            boolean saved = Imgcodecs.imwrite(diffImagePath, combined);
            if (saved) {
                System.out.println("❌ الصور مختلفة - تم حفظ صورة المقارنة في: " + diffImagePath);
            } else {
                System.out.println("❌ الصور مختلفة - فشل حفظ صورة المقارنة.");
            }
        }
        return isMatch;
    }
}
