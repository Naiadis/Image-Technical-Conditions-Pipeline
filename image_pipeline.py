"""
Libraries used during implementation.

  OpenCV:
      https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
    - Colourfulness (Hasler & Süsstrunk, 2003; PyImageSearch tutorial):
      https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    - imread (load image from file):
      https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    - cvtColor (BGR↔grayscale, BGR↔HSV):
      https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
    - split (separate channels):
      https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a

  NumPy:
    - mean, std, sqrt, abs, argmax, clip, histogram:
      https://numpy.org/doc/stable/reference/routines.math.html
      https://numpy.org/doc/stable/reference/generated/numpy.mean.html
      https://numpy.org/doc/stable/reference/generated/numpy.std.html

  pandas:
    - DataFrame:
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    - to_csv:
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

  scikit-image (texture):
      https://www.tutorialspoint.com/scikit-image/scikit-image-glcm-texture-features.htm
    - graycomatrix:
      https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
    - graycoprops:
      https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops

  Outlier detection (MAD method):
    Shimizu Y (2022) Multiple Desirable Methods in Outlier Detection of Univariate
    Data With R Source Codes. Front. Psychol. 12:819854.
    doi: 10.3389/fpsyg.2021.819854
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8801745/


What this script does so far:
1. Looks in a folder for images (jpg, jpeg, png).
2. For each image, it measures all 17 features from our plan (Maros et al., 2019):
   - Technical: Sharpness, Exposure, Contrast, Average Intensity
   - Colour: Colourfulness, Avg Saturation, Avg Hue, Dominant Hue, HSV Depth,
             Central H, Central S, Central V
   - Texture: Hue Texture, Saturation Texture
   - Composition: Height, Width, Sum of Height and Width
3. Saves all measurements into a CSV file (open in Excel / SPSS).

Later steps ='): image transformation, z-scoring maybe.
To do: - Write the exact version of the libraries used in the script (freezing dependencies).
    - Feature quality checks (e.g. saturation texture is always 0, or i get a weird value for colorfulness).
    - check for things like "double files", coroupt files, etc.
"""

import os
from typing import List, Dict, Any

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import pandas as pd  # pandas for table-like data (DataFrame)
from skimage.feature import graycomatrix, graycoprops  # Texture (GLCM)


# BASIC SETTINGS

# TODO: change this folder later to where we store our input images.
# For now, "data/input" folder is inside the repo and we drop test images there.
INPUT_DIR = "data/input"

# TODO: change this folder later if we want processed images somewhere else (dropbox or smth).
# In this first version we only SAVE A CSV here, not new images yet.
OUTPUT_DIR = "data/output"

CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "image_features_all.csv")
OUTLIER_REPORT_PATH = os.path.join(OUTPUT_DIR, "outlier_report.csv")

def ensure_output_folder_exists() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_image_files(input_dir: str) -> List[str]:
    """Find .jpg, .jpeg, .png files in the top level of input_dir."""
    if not os.path.isdir(input_dir):
        return []
    allowed = {".jpg", ".jpeg", ".png"}
    paths = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in allowed:
            paths.append(p)
    return paths


def _compute_glcm_texture(channel: np.ndarray, levels: int = 8) -> float:
    # Compute GLCM dissimilarity for a single channel (for texture).
    # Used for Hue Texture and Saturation Texture.
    
    # Ensure we have a valid range for graycomatrix.
    if channel.max() == channel.min():
        return 0.0 
    chan_min, chan_max = channel.min(), channel.max()
    quantized = np.clip(
        ((channel.astype(float) - chan_min) / (chan_max - chan_min + 1e-8) * (levels - 1)),
        0,
        levels - 1,
    ).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[5],
        angles=[0],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    # graycoprops returns shape (1, 1) for one distance and one angle
    return float(graycoprops(glcm, "dissimilarity")[0, 0])


def compute_all_features_for_image(image_bgr: np.ndarray, image_path: str) -> Dict[str, Any]:

    # Composition Categorie (Height, Width and Sum of Height and Width)
    height, width = image_bgr.shape[:2]
    sum_hw = height + width

    # Convert to grayscale for technical features
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

 
    # Technical Categorie (Sharpness, Exposure, Contrast, Average Intensity)
    # Sharpness: Laplacian variance – higher = sharper (more edges)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Exposure and Average Intensity: same thing (mean of grayscale)
    exposure = float(np.mean(image_gray))
    avg_intensity = exposure

    # Contrast: standard deviation of grayscale intensity
    contrast = float(np.std(image_gray))


    # Colour Categorie (Colourfulness, Average saturation and hue , Dominant Hue, HSV Depth, Central HSV metrics )
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(image_hsv)

    # Average saturation and hue
    avg_sat = float(np.mean(s_channel))
    avg_hue = float(np.mean(h_channel))

    # Colourfulness
    (B, G, R) = cv2.split(image_bgr.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rg_mean, rg_std) = (np.mean(rg), np.std(rg))
    (yb_mean, yb_std) = (np.mean(yb), np.std(yb))
    std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
    colorfulness = float(std_root + (0.3 * mean_root))

    # Dominant hue: most frequent hue bin (OpenCV H is 0–179, 180 bins)
    hue_hist, _ = np.histogram(h_channel.ravel(), bins=180, range=(0, 180))
    dom_hue = float(np.argmax(hue_hist))

    # HSV Depth: std of V channel (brightness variation)
    hsv_depth = float(np.std(v_channel))

    # Central HSV: mean of central 50% crop (middle 50% in height and width)
    h_mid, w_mid = height // 2, width // 2
    h_quarter, w_quarter = height // 4, width // 4
    center_h = h_channel[h_quarter : h_quarter + h_mid, w_quarter : w_quarter + w_mid]
    center_s = s_channel[h_quarter : h_quarter + h_mid, w_quarter : w_quarter + w_mid]
    center_v = v_channel[h_quarter : h_quarter + h_mid, w_quarter : w_quarter + w_mid]
    central_h = float(np.mean(center_h))
    central_s = float(np.mean(center_s))
    central_v = float(np.mean(center_v))


    # Texture Categorie (Hue Texture, Saturation Texture)
    hue_tex = _compute_glcm_texture(h_channel, levels=8)
    sat_tex = _compute_glcm_texture(s_channel, levels=8)


    # Feature dictionary
    features: Dict[str, Any] = {
        "filename": os.path.basename(image_path),
        # Technical
        "sharpness": sharpness,
        "exposure": exposure,
        "contrast": contrast,
        "avg_intensity": avg_intensity,
        # Colour
        "colorfulness": colorfulness,
        "avg_sat": avg_sat,
        "avg_hue": avg_hue,
        "dom_hue": dom_hue,
        "hsv_depth": hsv_depth,
        "central_h": central_h,
        "central_s": central_s,
        "central_v": central_v,
        # Texture
        "hue_tex": hue_tex,
        "sat_tex": sat_tex,
        # Composition
        "height": float(height),
        "width": float(width),
        "sum_hw": sum_hw,
    }
    return features


def process_all_images(input_dir: str) -> pd.DataFrame:
    image_paths = list_image_files(input_dir)

    if not image_paths:
        print(f"No images found in folder: {input_dir}")
        print("TIP: Place some .jpg/.jpeg/.png files in that folder and run again.")
        return pd.DataFrame()  # empty table

    all_feature_rows: List[Dict[str, float]] = []

    for path in image_paths:
        print(f"Processing image: {path}")
        image = cv2.imread(path)

        if image is None:
            # If OpenCV could not read the image, skip it with a clear message.
            print(f"WARNING: Could not read image at path: {path}. Skipping.")
            continue

        features = compute_all_features_for_image(image, path)
        all_feature_rows.append(features)

    if not all_feature_rows:
        print("No valid images were processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_feature_rows)
    return df


def _compute_mad(x: np.ndarray) -> float:
    # Median Absolute Deviation [MAD = 1.4826 × Med(|x - Med(x)|)].
    med = np.median(x)
    raw_mad = np.median(np.abs(x - med))
    return 1.4826 * raw_mad


def flag_mad_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Flag images whose feature values fall outside median ± 2.5 × MAD.
    numeric_cols = [c for c in df.columns if c != "filename"]
    rows = []
    for feat in numeric_cols:
        vals = df[feat].values
        median = np.median(vals)
        mad = _compute_mad(vals)
        k = 2.5
        lower = median - k * mad
        upper = median + k * mad
        for i, fn in enumerate(df["filename"]):
            v = vals[i]
            out = v < lower or v > upper
            rows.append({
                "filename": fn,
                "feature": feat,
                "value": v,
                "median": median,
                "mad": mad,
                "lower_bound": lower,
                "upper_bound": upper,
                "is_outlier": out,
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Image feature extraction (all 17 features)")
    print(f"Looking for images in: {INPUT_DIR}")

    ensure_output_folder_exists()

    df = process_all_images(INPUT_DIR)

    if df.empty:
        print("No data to save (no images were found or processed).")
        return

    # Save features to CSV
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to: {CSV_OUTPUT_PATH}")

    # Outlier flagging
    print("\n Outlier detection")
    outlier_df = flag_mad_outliers(df)
    outlier_df.to_csv(OUTLIER_REPORT_PATH, index=False)
    print(f"Outlier report saved to: {OUTLIER_REPORT_PATH}")

    # Summary: total outliers
    n_outliers = outlier_df["is_outlier"].sum()
    print(f"Total outlier values: {n_outliers}")


if __name__ == "__main__":
    # This makes sure main() runs only when you run this file directly,
    # not when you import it from somewhere else.
    main()

