package io.github.ahmedraafat2.imagecomparator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.IOException;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class ImageComparatorTest {
    @TempDir Path temp;
    private final ImageComparator comparator = new ImageComparator();

    @Test void identicalImagesMatch() throws Exception {
        Path a = image("a.png", 12, 12);
        Path b = image("b.png", 12, 12);
        ImageComparator.ComparisonResult result = comparator.compare(a, b);
        assertTrue(result.matches());
        assertEquals(0, result.differentPixels());
    }

    @Test void onePixelChangeIsCountedAndDiagnosticHasThreePanels() throws Exception {
        Path a = image("a.png", 12, 12);
        Path b = image("b.png", 12, 12);
        Mat changed = Imgcodecs.imread(b.toString());
        try {
            changed.put(5, 5, new byte[] {(byte) 255, 0, 0});
            assertTrue(Imgcodecs.imwrite(b.toString(), changed));
        } finally { changed.release(); }
        Path output = temp.resolve("nested/diff.png");
        ImageComparator.ComparisonResult result = comparator.compare(a, b, output,
                ImageComparator.ComparisonOptions.defaults());
        assertFalse(result.matches());
        assertEquals(1, result.differentPixels());
        Mat diagnostic = Imgcodecs.imread(output.toString());
        try {
            assertEquals(36, diagnostic.cols());
            assertEquals(12, diagnostic.rows());
        } finally { diagnostic.release(); }
    }

    @Test void readsCommonImageFormats() throws Exception {
        Path png = image("reference.png", 12, 12);
        for (String extension : new String[] {"jpg", "bmp"}) {
            Path other = image("actual." + extension, 12, 12);
            assertTrue(comparator.compare(png, other).matches(), extension);
        }
    }
    @Test void unreadableImagesFailInsteadOfMatchingBlankPlaceholders() {
        assertThrows(IOException.class, () -> comparator.compare(
                temp.resolve("missing-a.png"), temp.resolve("missing-b.png")));
    }

    @Test void differentDimensionsFailByDefault() throws Exception {
        Path a = image("a.png", 12, 12);
        Path b = image("b.png", 13, 12);
        assertThrows(IllegalArgumentException.class, () -> comparator.compare(a, b));
    }

    @Test void resizePoliciesCompareAtTheirDocumentedDimensions() throws Exception {
        Path reference = image("reference.png", 12, 10);
        Path actual = image("actual.png", 8, 14);
        ImageComparator.ComparisonResult result = comparator.compare(reference, actual,
                ImageComparator.ComparisonOptions.builder()
                        .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.RESIZE_BOTH_TO_MINIMUM)
                        .build());
        assertEquals(8, result.width());
        assertEquals(10, result.height());

        ImageComparator.ComparisonResult referenceSized = comparator.compare(reference, actual,
                ImageComparator.ComparisonOptions.builder()
                        .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.RESIZE_ACTUAL_TO_REFERENCE)
                        .build());
        assertEquals(12, referenceSized.width());
        assertEquals(10, referenceSized.height());
    }

    private Path image(String name, int width, int height) {
        Path path = temp.resolve(name);
        Mat image = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));
        try { assertTrue(Imgcodecs.imwrite(path.toString(), image)); }
        finally { image.release(); }
        return path;
    }
}
