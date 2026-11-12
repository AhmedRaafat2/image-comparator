# Image Comparator

Java library for pixel-level image comparison. It reports changed-pixel counts and percentages, and can save a diagnostic image with three panels: reference, actual image marked with red circles around changed regions, and grayscale difference.

## Requirements

- Java 8 or newer
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

The diagnostic image is written only when the allowed difference is exceeded. Its size is three times the input width by the input height.

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

- compare(reference, actual) returns a result without writing an image.
- compare(reference, actual, options) compares with custom thresholds and policies.
- compare(reference, actual, output, options) also writes the three-panel diagnostic on mismatch.
- ComparisonResult provides matches(), differentPixels(), totalPixels(), differencePercent(), dimensions, and the diagnostic path.

The original default-package methods compareAndSaveDiffIfNotMatch(...) and areImagesSimilar(...) remain deprecated adapters for source compatibility. New code should use the packaged API above.

## Image formats

Images are decoded and encoded by OpenCV. Available formats depend on the codecs in the OpenCV distribution and the file extension used for output. PNG, JPEG, and BMP are covered by the project tests for the bundled distribution; other codecs can vary. Unsupported or missing inputs fail with an exception rather than being treated as matching blank images.

## Build

~~~shell
mvn test
~~~
## Publish a release

The release workflow is started manually; it does not publish on every push or tag.

1. Push the code you want to release to the branch you intend to release from.
2. In GitHub, open **Actions** → **Release to GitHub & Maven Central** → **Run workflow**.
3. Select that branch and enter a version such as 1.1.0 without the v.
4. Run the workflow.

The workflow validates the version and checks that its vVERSION tag is unused, updates the Maven version, runs tests and creates source/Javadoc jars, then commits the version change to the selected branch. It signs the Maven artifacts, uploads them to Sonatype Central with automatic publishing, waits for the deployment to reach PUBLISHED, and then creates the matching GitHub Release and tag with the artifacts attached.

Configure these repository Actions secrets before the first release:

| Secret | Value |
| --- | --- |
| GPG_PRIVATE_KEY | Armored private signing key |
| GPG_PASSPHRASE | Passphrase for that private key |
| MAVEN_CENTRAL_USERNAME | Username part of a Central Portal user token |
| MAVEN_CENTRAL_TOKEN | Password part of that user token |

If an upload has reached Maven Central but a later step fails, check the Central Portal before rerunning; Maven Central does not allow reusing a published version.

The Maven Central namespace must be verified and the public GPG key must be available from a key server. Versions already published to Maven Central are immutable; use a new version for every release. If the workflow fails after the version commit, fix the cause and rerun the same version while its Git tag is still unused.