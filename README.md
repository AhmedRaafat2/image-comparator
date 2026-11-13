# Image Comparator

Java library for pixel-level image comparison. It reports changed-pixel counts and percentages, and can save a diagnostic image with three panels: reference, actual image marked with red circles around changed regions, and grayscale difference.

## Requirements

- Java 17 or newer
- Maven

## Add the dependency

Maven Central (replace RELEASE_VERSION with the latest release):

~~~xml
<dependency>
  <groupId>io.github.ahmedraafat2</groupId>
  <artifactId>image-comparator</artifactId>
  <version>RELEASE_VERSION</version>
</dependency>
~~~

Gradle:

~~~gradle
dependencies {
    implementation 'io.github.ahmedraafat2:image-comparator:RELEASE_VERSION'
}
~~~

## Quick start

~~~java
import io.github.ahmedraafat2.imagecomparator.ImageComparator;
import java.nio.file.Paths;

ImageComparator comparator = new ImageComparator();
ImageComparator.ComparisonResult result = comparator.compare(
    Paths.get("reference.png"), Paths.get("actual.png"),
    Paths.get("build/reports/image-diff.png"),
    ImageComparator.ComparisonOptions.defaults()
);
System.out.println("Match: " + result.matches());
System.out.println("Changed pixels: " + result.differentPixels());
System.out.println("Difference: " + result.differencePercent() + "%");
~~~

The diagnostic image is written when the allowed difference is exceeded or either input is missing. It contains three panels: reference, current image with red marks, and a grayscale difference. Its size is three times the input width by the input height.

Both inputs must use the same filename extension. Dimensions must match by default; the resize policies are explicit opt-ins. If an input cannot be decoded, the diagnostic uses a white panel labeled `Image not found` in its place (two such panels if both inputs are missing). When a diagnostic is written, the library also writes a reduced JPEG beside it using the `-compressed.jpg` suffix, at up to 1200 pixels wide and JPEG quality 80. `ComparisonResult.compressedDiagnosticImage()` returns that path.

## Compare images and save a report

This example writes a three-panel diagnostic and a reduced JPEG when the images differ or an input is missing. The input paths must use the same file extension. By default, their pixel dimensions must also match.

~~~java
import io.github.ahmedraafat2.imagecomparator.ImageComparator;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CompareImages {
    public static void main(String[] args) throws IOException {
        Path reference = Paths.get("screenshots/reference.png");
        Path actual = Paths.get("screenshots/current.png");
        Path diagnostic = Paths.get("build/reports/page-diff.png");

        ImageComparator comparator = new ImageComparator();
        ImageComparator.ComparisonOptions options =
                ImageComparator.ComparisonOptions.builder()
                        .allowedDifferencePercent(0.02)
                        .pixelThreshold(0)
                        .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.FAIL)
                        .build();

        ImageComparator.ComparisonResult result =
                comparator.compare(reference, actual, diagnostic, options);

        if (result.matches()) {
            System.out.println("Images match within the allowed difference.");
        } else {
            System.out.println("Images differ.");
            System.out.println("Diagnostic: " + result.diagnosticImage());
            System.out.println("Reduced JPEG: " + result.compressedDiagnosticImage());
            System.out.println("Difference: " + result.differencePercent() + "%");
        }
    }
}
~~~

## Resize an image for reports

`ImageResizer` reduces or enlarges an image to a target width while keeping its aspect ratio. It creates missing output directories and throws an `IOException` if the input cannot be read or output cannot be written.

~~~java
import io.github.ahmedraafat2.imagecomparator.ImageResizer;
import java.nio.file.Paths;

ImageResizer.resizeToWidth(
    Paths.get("screenshots/full.png"),
    Paths.get("reports/preview.png"),
    800
);
~~~

## Configure the comparison

~~~java
ImageComparator.ComparisonOptions options = ImageComparator.ComparisonOptions.builder()
    .allowedDifferencePercent(0.02) // accept up to 0.02% changed pixels
    .pixelThreshold(0)              // any channel difference greater than this value counts
    .minimumRegionArea(0)            // keep tiny marks such as dots and diacritics
    .circleThickness(2)
    .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.FAIL)
    .build();

ImageComparator.ComparisonResult result = comparator.compare(
    Paths.get("reference.png"), Paths.get("actual.png"),
    Paths.get("build/reports/diff.png"), options
);
~~~

pixelThreshold applies independently to each BGR channel. A pixel counts as changed when any channel differs by more than the threshold. The default is zero, so even a one-level channel change counts. The default size policy is FAIL, avoiding false differences caused by silently stretching pages. For intentional resizing, choose RESIZE_ACTUAL_TO_REFERENCE or RESIZE_BOTH_TO_MINIMUM. The deprecated areImagesSimilar adapter preserves its former minimum-dimension resize behavior.

## API

- compare(reference, actual) returns a result without writing files. Pass a diagnostic output path to create the three-panel image and its compressed copy.
- compare(reference, actual, options) compares with custom thresholds and policies.
- compare(reference, actual, output, options) also writes the three-panel diagnostic on mismatch.
- ComparisonResult provides matches(), differentPixels(), totalPixels(), differencePercent(), dimensions, and the diagnostic path.

The original default-package methods compareAndSaveDiffIfNotMatch(...) and areImagesSimilar(...) remain deprecated adapters for source compatibility. New code should use the packaged API above.

## Image formats

Images are decoded and encoded by OpenCV. Available formats depend on the codecs in the OpenCV distribution and the file extension used for output. PNG, JPEG, and BMP are covered by the project tests for the bundled distribution; other codecs can vary. Unsupported or missing inputs are treated as missing images and make the comparison fail. If a diagnostic output path is supplied, the missing panel is shown in white with the label `Image not found`.

## Build

~~~shell
mvn test
~~~