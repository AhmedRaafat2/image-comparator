# Image Comparator

Java 17+ library for pixel-level image comparison. It reports changed-pixel counts and percentages, and can save a diagnostic image with three panels: the reference, the actual image marked with red circles around changed regions, and a grayscale difference image.

## Requirements

- Java 17 or newer
- Maven (for Maven projects)

## Add the dependency

### Maven

```xml
<dependency>
    <groupId>io.github.ahmedraafat2</groupId>
    <artifactId>image-comparator</artifactId>
    <version>1.1.0</version>
</dependency>
```

### Gradle

```gradle
dependencies {
    implementation 'io.github.ahmedraafat2:image-comparator:1.1.0'
}
```

## Compare two images

This example compares a baseline with a newly captured screenshot. It writes the three-panel report and a smaller JPEG when the images differ or an input is missing.

```java
import io.github.ahmedraafat2.imagecomparator.ImageComparator;

import java.io.IOException;
import java.nio.file.Path;

public class CompareImages {
    public static void main(String[] args) throws IOException {
        Path reference = Path.of("screenshots/reference.png");
        Path actual = Path.of("screenshots/current.png");
        Path report = Path.of("build/reports/page-diff.png");

        ImageComparator comparator = new ImageComparator();
        ImageComparator.ComparisonOptions options =
                ImageComparator.ComparisonOptions.builder()
                        .allowedDifferencePercent(0.02)
                        .pixelThreshold(0)
                        .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.FAIL)
                        .build();

        ImageComparator.ComparisonResult result =
                comparator.compare(reference, actual, report, options);

        System.out.println("Images match: " + result.matches());
        System.out.println("Changed pixels: " + result.differentPixels());
        System.out.println("Difference: " + result.differencePercent() + "%");
        if (result.diagnosticImage() != null) {
            System.out.println("Diagnostic: " + result.diagnosticImage());
            System.out.println("Smaller JPEG: " + result.compressedDiagnosticImage());
        }
    }
}
```

### Parameters in the comparison example

| Parameter | What it controls |
| --- | --- |
| `reference` | Path to the baseline image that the actual image is compared against. |
| `actual` | Path to the new image being checked. |
| `report` | Where to write the three-panel diagnostic image on a mismatch or missing input. Its parent folders are created if needed. |
| `allowedDifferencePercent(0.02)` | Maximum percentage of changed pixels that still counts as a match. `0.02` means 0.02 percent. The default is `0.0`. |
| `pixelThreshold(0)` | Per-channel BGR difference required for a pixel to count as changed. A pixel is marked when any channel differs by more than this value. Range: 0–255; default: `0`. |
| `sizeMismatchPolicy(FAIL)` | What to do when image dimensions differ. `FAIL` throws an exception; other supported policies are described below. |

The report is composed of three panels: reference, actual with red circles around changed regions, and grayscale pixel differences. The library also writes a smaller JPEG beside it with the `-compressed.jpg` suffix, at up to 1200 pixels wide and JPEG quality 80. `compressedDiagnosticImage()` returns that JPEG path.

Both input paths must have the same file extension. By default, image dimensions must also match. If an input is missing or cannot be decoded, its report panel is white and labeled `Image not found`. If both inputs are missing, both panels show that message.

## Compare two directories

`ImageBatchComparator` walks both directory trees recursively and pairs images by relative path and filename (without the extension). It writes a diagnostic for each different or missing image and a `summary.json` report.

```java
import io.github.ahmedraafat2.imagecomparator.BatchComparisonResult;
import io.github.ahmedraafat2.imagecomparator.ImageBatchComparator;
import io.github.ahmedraafat2.imagecomparator.ImageComparator;

import java.io.IOException;
import java.nio.file.Path;

public class CompareDirectories {
    public static void main(String[] args) throws IOException {
        Path referenceDirectory = Path.of("screenshots/reference");
        Path actualDirectory = Path.of("screenshots/current");
        Path reportsDirectory = Path.of("build/reports/image-comparison");
        ImageComparator.ComparisonOptions options = ImageComparator.ComparisonOptions.defaults();

        BatchComparisonResult result = new ImageBatchComparator().compareDirectories(
                referenceDirectory, actualDirectory, reportsDirectory, options);

        System.out.printf("Compared %d image paths: %d matched, %d different.%n",
                result.totalFiles(), result.matchedCount(), result.differentCount());
        System.out.println("Summary: " + result.summaryFile());
    }
}
```

| Parameter | What it controls |
| --- | --- |
| `referenceDirectory` | Root folder containing baseline images. |
| `actualDirectory` | Root folder containing images to compare against the baseline. |
| `reportsDirectory` | Destination for per-image diagnostics and `summary.json`. It must be outside both input trees. |
| `options` | Pixel thresholds and size mismatch behavior. Use `ComparisonOptions.defaults()` for exact pixel comparison and fail-on-size-mismatch behavior. |

Supported extensions are PNG, JPG/JPEG, BMP, TIF/TIFF, and WEBP. Other files are ignored. If both trees contain the same relative filename with different extensions, the item is reported as an extension mismatch. If multiple files share a relative filename without extensions, the item is marked ambiguous. Missing files, size mismatches, and processing errors are recorded in the result and JSON summary.

## Tune comparison behavior

```java
ImageComparator.ComparisonOptions options = ImageComparator.ComparisonOptions.builder()
        .allowedDifferencePercent(0.02)
        .pixelThreshold(3)
        .minimumRegionArea(4)
        .circleThickness(2)
        .sizeMismatchPolicy(ImageComparator.SizeMismatchPolicy.RESIZE_ACTUAL_TO_REFERENCE)
        .build();
```

| Option | Meaning | Default |
| --- | --- | --- |
| `allowedDifferencePercent(double)` | Acceptable changed-pixel percentage, from 0 to 100. A comparison matches when its measured percentage is at or below this value. | `0.0` |
| `pixelThreshold(double)` | Per-channel BGR threshold, from 0 to 255. A pixel changes if any channel difference is greater than the threshold. | `0.0` |
| `minimumRegionArea(double)` | Minimum contour area, in pixels, to draw as a red circle in the diagnostic. Smaller regions are omitted from the markup; they still count toward the difference metrics. | `0.0` |
| `circleThickness(int)` | Thickness of red circles drawn around changed regions. Must be at least 1. | `2` |
| `sizeMismatchPolicy(SizeMismatchPolicy)` | Behavior when image widths or heights differ. | `FAIL` |

Available size policies:

- `FAIL`: throw `IllegalArgumentException` if dimensions differ.
- `RESIZE_ACTUAL_TO_REFERENCE`: resize the actual image to the reference dimensions before comparing.
- `RESIZE_BOTH_TO_MINIMUM`: resize both images to the smaller width and smaller height before comparing.

## Resize an image

`ImageResizer.resizeToWidth` keeps the original aspect ratio. For JPEG output, an optional quality value controls compression.

```java
import io.github.ahmedraafat2.imagecomparator.ImageResizer;

import java.io.IOException;
import java.nio.file.Path;

public class ResizeImage {
    public static void main(String[] args) throws IOException {
        ImageResizer.resizeToWidth(
                Path.of("screenshots/full.png"),
                Path.of("reports/preview.jpg"),
                800,
                85
        );
    }
}
```

| Parameter | What it controls |
| --- | --- |
| First `Path` | Input image to resize. |
| Second `Path` | Destination file. The extension selects the output format; parent folders are created if needed. |
| `800` | Target width in pixels. Height is calculated to preserve the aspect ratio. Must be positive. |
| `85` | Optional JPEG quality from 0 (smaller file, lower quality) to 100 (larger file, higher quality). It applies only to `.jpg`/`.jpeg` output. If omitted, the library uses `85`. |

The output path may be the same as the input path to replace the image. The path-based API decodes images as BGR, so it does not preserve alpha transparency. Shrinking uses area interpolation; enlarging uses cubic interpolation.

## Results and overloads

- `compare(reference, actual)` compares with default options and does not write a report.
- `compare(reference, actual, options)` compares with the provided options and does not write a report.
- `compare(reference, actual, report, options)` also writes the three-panel report and smaller JPEG when the comparison differs or an input is missing.
- `ImageBatchComparator.compareDirectories(referenceDirectory, actualDirectory, reportsDirectory, options)` recursively compares image trees and writes `summary.json`.
- `ComparisonResult.matches()` is `true` only when both images exist and the difference is within the allowed percentage.
- `differentPixels()`, `totalPixels()`, `differencePercent()`, `width()`, and `height()` return comparison metrics.
- `diagnosticImage()` and `compressedDiagnosticImage()` return output paths, or `null` when no report was created.

## Image formats

Images are decoded and encoded by OpenCV. Supported formats depend on the codecs in the OpenCV distribution. PNG, JPEG, and BMP are covered by the project tests for the bundled distribution; other codecs may vary. Input extensions must match. Unsupported or missing inputs are treated as missing images.

## Build from source

```shell
mvn clean verify
```
