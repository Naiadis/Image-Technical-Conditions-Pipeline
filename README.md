# Image Technical Conditions Pipeline

Image standardisation pipeline for controlling technical image features
across experimental conditions (isolated, in-use, environmental) in a
multivendor e-commerce study on trust and diagnosticity.

## Pipeline Steps

0. **Quality checks** -- corrupt files, duplicates, grayscale images, minimum resolution.
1. **Measure** 17 features on all original images (Maros et al., 2019):
   - Technical: Sharpness, Exposure, Contrast, Average Intensity
   - Colour: Colourfulness, Avg Saturation, Avg Hue, Dominant Hue,
     HSV Depth, Central H, Central S, Central V
   - Texture: Hue Texture, Saturation Texture
   - Composition: Height, Width, Sum of Height and Width
2. **Within-condition MAD outlier detection** (Shimizu, 2022):
   median +/- 2.5 x MAD, run separately per condition.
   Images with too many measure-only outliers are excluded.
3. **Find best-matching triplet** (1 image per condition) with lowest
   feature distance across all non-excluded images.
4. **Adjust** the chosen triplet toward their shared mean (damped):
   exposure, contrast, saturation, sharpness.
5. **Re-measure** the adjusted triplet and save before/after CSVs.

## How to Run

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (pinned versions for replicability)
pip install -r requirements.txt

# Run the pipeline
python image_pipeline.py
```

## Folder Structure

```
data/
  input/
    isolated/          # Product-only images (white background)
    in_use/            # Product being used by a person
    environmental/     # Product shown in a room/context
  output/
    image_features_all.csv     # Features measured on originals ("before")
    image_features_after.csv   # Features measured on adjusted triplet ("after")
    outlier_report.csv         # Full MAD outlier report
  adjusted_images/             # Adjusted copies of the chosen triplet
```

## Libraries and References

- **OpenCV** ([docs](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)):
  image loading, colour space conversion, channel manipulation, Gaussian blur.
  - Colourfulness metric: Hasler & Susstrunk (2003);
    [PyImageSearch tutorial](https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/)
  - `imread`: [docs](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
  - `cvtColor` (BGR to grayscale, BGR to HSV): [docs](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
  - `split` (separate channels): [docs](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a)

- **NumPy** ([docs](https://numpy.org/doc/stable/reference/routines.math.html)):
  mean, std, sqrt, abs, argmax, clip, histogram.

- **pandas** ([docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)):
  DataFrame, to_csv.

- **scikit-image** ([tutorial](https://www.tutorialspoint.com/scikit-image/scikit-image-glcm-texture-features.htm)):
  GLCM texture features.
  - `graycomatrix`: [docs](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix)
  - `graycoprops`: [docs](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops)

- **Outlier detection (MAD method)**:
  Shimizu Y (2022) Multiple Desirable Methods in Outlier Detection of Univariate
  Data With R Source Codes. *Front. Psychol.* 12:819854.
  doi: [10.3389/fpsyg.2021.819854](https://doi.org/10.3389/fpsyg.2021.819854)

- **Feature set based on**:
  Maros, A., Belém, F., Silva, R., Canuto, S., Almeida, J. M., & Gonçalves, M. A. (2019). Image Aesthetics and its Effects on Product Clicks in E-Commerce Search. https://www.elo7.com.br