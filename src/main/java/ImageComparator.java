import io.github.ahmedraafat2.imagecomparator.ImageComparator.ComparisonOptions;
import io.github.ahmedraafat2.imagecomparator.ImageComparator.ComparisonResult;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * Backward-compatible adapter for the original default-package API.
 * Prefer io.github.ahmedraafat2.imagecomparator.ImageComparator.
 */
@Deprecated
public class ImageComparator {
    private final io.github.ahmedraafat2.imagecomparator.ImageComparator delegate =
            new io.github.ahmedraafat2.imagecomparator.ImageComparator();

    /**
     * @param referencePath baseline file
     * @param actualPath current file
     * @param outputPath path for the diagnostic image
     * @param allowedDifferencePercent accepted changed-pixel percentage
     * @return whether the images match the threshold
     * @deprecated Use the packaged compare API. */
    @Deprecated
    public boolean compareAndSaveDiffIfNotMatch(String referencePath, String actualPath,
                                                 String outputPath, double allowedDifferencePercent) {
        try {
            ComparisonOptions options = ComparisonOptions.builder()
                    .allowedDifferencePercent(allowedDifferencePercent)
                    .sizeMismatchPolicy(io.github.ahmedraafat2.imagecomparator.ImageComparator.SizeMismatchPolicy.RESIZE_ACTUAL_TO_REFERENCE).build();
            ComparisonResult result = delegate.compare(Paths.get(referencePath), Paths.get(actualPath),
                    Paths.get(outputPath), options);
            return result.matches();
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not compare images", e);
        }
    }

    /**
     * @param referencePath baseline file
     * @param actualPath current file
     * @param similarityThreshold minimum similarity in the range 0..1
     * @return whether the images match the threshold
     * @deprecated Use the packaged compare API. */
    @Deprecated
    public boolean areImagesSimilar(String referencePath, String actualPath, double similarityThreshold) {
        if (!Double.isFinite(similarityThreshold) || similarityThreshold < 0 || similarityThreshold > 1) {
            throw new IllegalArgumentException("similarityThreshold must be between 0 and 1");
        }
        try {
            ComparisonOptions options = ComparisonOptions.builder()
                    .allowedDifferencePercent((1.0 - similarityThreshold) * 100.0)
                    .sizeMismatchPolicy(io.github.ahmedraafat2.imagecomparator.ImageComparator.SizeMismatchPolicy.RESIZE_BOTH_TO_MINIMUM).build();
            return delegate.compare(Paths.get(referencePath), Paths.get(actualPath), options).matches();
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not compare images", e);
        }
    }
}