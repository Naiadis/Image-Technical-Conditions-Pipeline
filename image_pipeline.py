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
3. Saves results to image_features_all.csv (in case we want to open it in Excel / SPSS).
4. Flags outliers using MAD method (Shimizu, 2022): values outside median ± 2.5 × MAD.
5. Saves outlier report to outlier_report.csv.

Later steps ='): image transformation, z-scoring maybe.
To do: - Write the exact version of the libraries used in the script (freezing dependencies).
    - Feature quality checks (e.g. saturation texture is always 0, or i get a weird value for colorfulness).
    - check for things like "double files", coroupt files, etc.
"""

import os
import shutil
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
# After-adjustment features (for \"after\" comparison)
CSV_AFTER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "image_features_after.csv")

# Folder where we will save resized copies of the images.
# This keeps the original images untouched.
RESIZED_IMAGE_DIR = os.path.join("data", "resized_images")

# Folder where we will save adjusted images (after changing brightness, contrast, etc.).
ADJUSTED_IMAGE_DIR = os.path.join("data", "adjusted_images")

# Feature separation for later transformations
# These lists mark which measured features we plan to actively adjust
# (for example via brightness/contrast/saturation/resize) and which ones
# we will mostly monitor and, if needed, use for excluding images.

ADJUSTABLE_FEATURES = [
    "exposure",
    "avg_intensity",
    "contrast",
    "avg_sat",
    "height",
    "width",
]

MEASURE_ONLY_FEATURES = [
    "sharpness",
    # Colour properties we usually do not change directly
    "colorfulness",
    "avg_hue",
    "dom_hue",
    "hsv_depth",
    "central_h",
    "central_s",
    "central_v",
    # Texture (GLCM-based)
    "hue_tex",
    "sat_tex",
    # Derived size measure (follows from height and width)
    "sum_hw",
]


def ensure_output_folder_exists() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_resized_folder_exists() -> None:
    """
    Make sure the folder for resized images exists.
    We keep resized copies separate so original images are not changed.
    """

    os.makedirs(RESIZED_IMAGE_DIR, exist_ok=True)


def ensure_adjusted_folder_exists() -> None:
    """
    Make sure the folder for adjusted images exists.
    We keep adjusted copies separate so original images are not changed.
    """

    os.makedirs(ADJUSTED_IMAGE_DIR, exist_ok=True)


def reset_adjusted_folder() -> None:
    """
    Clear the adjusted images folder so each run starts from originals.
    This avoids stacking the same adjustment multiple times across runs.
    Within a single run, adjustments are still applied in sequence
    (exposure, then contrast, then saturation).
    """

    if os.path.isdir(ADJUSTED_IMAGE_DIR):
        shutil.rmtree(ADJUSTED_IMAGE_DIR)
    os.makedirs(ADJUSTED_IMAGE_DIR, exist_ok=True)


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

    # Average Intensity: mean of grayscale values
    avg_intensity = float(np.mean(image_gray))

    # Contrast: standard deviation of grayscale intensity
    contrast = float(np.std(image_gray))


    # Colour Categorie (Colourfulness, Average saturation and hue , Dominant Hue, HSV Depth, Central HSV metrics )
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(image_hsv)

    # Exposure: mean of V channel (brightness in HSV space)
    exposure = float(np.mean(v_channel))

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


def resize_and_save_all_images(
    input_dir: str,
    output_dir: str,
    target_width: int = 800,
    target_height: int = 800,
) -> None:
    """
    Resize all images from input_dir, preserving aspect ratio, and save copies.

    - Images are scaled so that they fit *within* target_width x target_height.
    - Aspect ratio is preserved (no stretching or squashing).
    - We do NOT overwrite the original files.
    """

    if not os.path.isdir(input_dir):
        print(f"Cannot resize images: input folder does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_paths = list_image_files(input_dir)
    if not image_paths:
        print(f"No images found in folder for resizing: {input_dir}")
        return

    print(
        f"\nResizing images to fit within {target_width}x{target_height} "
        f"(preserving aspect ratio) and saving copies to: {output_dir}"
    )

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"WARNING: Could not read image at path (resize step): {path}. Skipping.")
            continue

        # Current size of the image
        h, w = image.shape[:2]

        # If the image already fits within the target box, keep it as-is.
        if h <= target_height and w <= target_width:
            resized = image
        else:
            # Scale image to fit within target_width x target_height
            scale = min(target_width / float(w), target_height / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        # Keep only the filename part when saving.
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(save_path, resized)
        if not success:
            print(f"WARNING: Could not save resized image to: {save_path}")

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


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print mean and standard deviation for adjustable features."""

    print("\nFeature summary for adjustable features (all images):")
    for feat in ADJUSTABLE_FEATURES:
        if feat in df.columns:
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            print(f"  {feat:>12}: mean = {mean_val:.2f}, std = {std_val:.2f}")


def summarize_outliers_by_image(outlier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise how many outliers each image has, split by feature type.

    This will help later when we decide which images to exclude
    (e.g. many measure-only outliers) versus which ones we might
    try to adjust (adjustable-feature outliers).
    """

    def _feature_type(feat: str) -> str:
        if feat in ADJUSTABLE_FEATURES:
            return "adjustable"
        if feat in MEASURE_ONLY_FEATURES:
            return "measure_only"
        return "other"

    # Work on a copy so we don't modify the original DataFrame by accident.
    outlier_df = outlier_df.copy()
    outlier_df["feature_type"] = outlier_df["feature"].apply(_feature_type)

    # We only want rows where the value was actually flagged as an outlier.
    out_only = outlier_df[outlier_df["is_outlier"] == True]

    if out_only.empty:
        print("\nNo outliers to summarise (no values flagged).")
        return pd.DataFrame()

    # Count how many outliers each image has, for each feature type.
    summary = (
        out_only
        .groupby(["filename", "feature_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Make sure the expected columns exist even if there were no outliers
    # of that type in this particular run.
    for col in ["adjustable", "measure_only", "other"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["total_outliers"] = (
        summary["adjustable"] + summary["measure_only"] + summary["other"]
    )

    print("\nOutlier summary per image:")
    for _, row in summary.iterrows():
        print(
            f"  {row['filename']}: "
            f"adjustable={row['adjustable']}, "
            f"measure_only={row['measure_only']}, "
            f"other={row['other']}, "
            f"total={row['total_outliers']}"
        )

    return summary


def build_adjustment_plan(
    features_df: pd.DataFrame, outlier_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a simple plan showing which adjustable features should be changed.

    - We only consider rows where:
        * the feature is in ADJUSTABLE_FEATURES, and
        * is_outlier is True (flagged by the MAD method).
    - For each adjustable feature we also attach the global mean, which
      can be used later as a target level for adjustments.
    """

    # Compute global means for adjustable features (for later use as targets).
    means = {}
    for feat in ADJUSTABLE_FEATURES:
        if feat in features_df.columns:
            means[feat] = features_df[feat].mean()

    # Keep only adjustable features that are flagged as outliers.
    mask = (outlier_df["is_outlier"] == True) & (
        outlier_df["feature"].isin(ADJUSTABLE_FEATURES)
    )
    to_adjust = outlier_df[mask].copy()

    if to_adjust.empty:
        print("\nNo adjustable features were flagged as outliers.")
        return pd.DataFrame()

    # Attach target_mean for each feature (if available).
    to_adjust["target_mean"] = to_adjust["feature"].map(means)

    print("\nAdjustment plan (per image and adjustable feature):")
    for _, row in to_adjust.iterrows():
        fname = row["filename"]
        feat = row["feature"]
        value = row["value"]
        target = row["target_mean"]
        print(
            f"  {fname} → {feat}: current={value:.2f}, "
            f"target_mean={target:.2f}"
        )

    return to_adjust


def adjust_exposure_from_plan(
    plan_df: pd.DataFrame,
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Apply simple exposure (brightness) adjustments according to the plan.

    For now we:
    - Look only at rows where feature == "exposure".
    - Adjust the V channel in HSV so that the mean exposure
      moves toward the global target_mean for exposure.
    - Save adjusted copies in output_dir, one per original image.
    """

    # Keep only exposure rows from the plan.
    if plan_df is None or plan_df.empty:
        print("\nNo adjustment plan available, skipping exposure adjustment.")
        return

    exposure_rows = plan_df[plan_df["feature"] == "exposure"]
    if exposure_rows.empty:
        print("\nNo exposure outliers in the adjustment plan, nothing to change.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAdjusting exposure for {len(exposure_rows)} images.")

    for _, row in exposure_rows.iterrows():
        filename = row["filename"]
        current_val = float(row["value"])
        target_val = float(row["target_mean"])

        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Could not read image for exposure adjustment: {img_path}")
            continue

        # Convert to HSV to manipulate brightness (V channel).
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if current_val <= 0:
            # Avoid division by zero; in this unlikely case we skip adjustment.
            print(f"WARNING: Non-positive current exposure for {filename}, skipping.")
            adjusted_v = v
        else:
            # Scale V so that its mean moves toward the target mean.
            scale = target_val / current_val
            v_float = v.astype(np.float32) * scale
            adjusted_v = np.clip(v_float, 0, 255).astype(np.uint8)

        hsv_adjusted = cv2.merge([h, s, adjusted_v])
        adjusted_bgr = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

        save_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(save_path, adjusted_bgr)
        if not success:
            print(f"WARNING: Could not save adjusted image to: {save_path}")


def _load_image_preferring_adjusted(filename: str) -> np.ndarray:
    """
    Helper: try to load the already adjusted version first,
    otherwise fall back to the original in INPUT_DIR.
    """

    # Prefer the image from the adjusted folder if it exists (so that
    # multiple adjustments stack on top of each other).
    adjusted_path = os.path.join(ADJUSTED_IMAGE_DIR, filename)
    if os.path.isfile(adjusted_path):
        image = cv2.imread(adjusted_path)
        if image is not None:
            return image

    # Fallback to the original input folder.
    original_path = os.path.join(INPUT_DIR, filename)
    return cv2.imread(original_path)


def adjust_contrast_from_plan(
    plan_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Apply simple contrast adjustments according to the plan.

    For now we:
    - Look only at rows where feature == "contrast".
    - Scale pixel values around their channel means so that the
      grayscale contrast (standard deviation) moves toward the
      global target_mean for contrast.
    - Save adjusted copies in output_dir, one per original image.
    """

    if plan_df is None or plan_df.empty:
        print("\nNo adjustment plan available, skipping contrast adjustment.")
        return

    contrast_rows = plan_df[plan_df["feature"] == "contrast"]
    if contrast_rows.empty:
        print("\nNo contrast outliers in the adjustment plan, nothing to change.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAdjusting contrast for {len(contrast_rows)} images.")

    for _, row in contrast_rows.iterrows():
        filename = row["filename"]
        current_contrast = float(row["value"])
        target_contrast = float(row["target_mean"])

        image = _load_image_preferring_adjusted(filename)
        if image is None:
            print(f"WARNING: Could not read image for contrast adjustment: {filename}")
            continue

        # Work in grayscale to estimate current mean and contrast.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        std_gray = float(gray.std())

        if std_gray <= 0:
            print(f"WARNING: Non-positive current contrast for {filename}, skipping.")
            adjusted = image
        else:
            # Scale factor to move standard deviation toward the target.
            scale = target_contrast / std_gray

            # Apply the same scale to each BGR channel around its mean.
            img_float = image.astype(np.float32)
            # Compute per-channel means.
            means = img_float.reshape(-1, 3).mean(axis=0)

            # new = mean + scale * (old - mean)
            adjusted = img_float - means
            adjusted *= scale
            adjusted += means
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        save_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(save_path, adjusted)
        if not success:
            print(f"WARNING: Could not save contrast-adjusted image to: {save_path}")


def adjust_saturation_from_plan(
    plan_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Apply simple saturation (avg_sat) adjustments according to the plan.

    - We look only at rows where feature == "avg_sat".
    - We scale the S channel in HSV so that the mean saturation moves
      toward the global target_mean for avg_sat.
    - Adjustments are stacked on top of any previous exposure/contrast
      changes by reading from the adjusted_images folder when possible.
    """

    if plan_df is None or plan_df.empty:
        print("\nNo adjustment plan available, skipping saturation adjustment.")
        return

    sat_rows = plan_df[plan_df["feature"] == "avg_sat"]
    if sat_rows.empty:
        print("\nNo saturation outliers in the adjustment plan, nothing to change.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAdjusting saturation for {len(sat_rows)} images.")

    for _, row in sat_rows.iterrows():
        filename = row["filename"]
        current_sat = float(row["value"])
        target_sat = float(row["target_mean"])

        image = _load_image_preferring_adjusted(filename)
        if image is None:
            print(f"WARNING: Could not read image for saturation adjustment: {filename}")
            continue

        if current_sat <= 0:
            print(f"WARNING: Non-positive current avg_sat for {filename}, skipping.")
            adjusted_bgr = image
        else:
            # Convert to HSV to manipulate the S (saturation) channel.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # Scale factor based on target vs current mean saturation.
            scale = target_sat / current_sat
            s_float = s.astype(np.float32) * scale
            adjusted_s = np.clip(s_float, 0, 255).astype(np.uint8)

            hsv_adjusted = cv2.merge([h, adjusted_s, v])
            adjusted_bgr = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

        save_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(save_path, adjusted_bgr)
        if not success:
            print(f"WARNING: Could not save saturation-adjusted image to: {save_path}")


def measure_adjusted_images() -> None:
    """
    Re-measure all features on the adjusted images and save to a new CSV.

    This gives us an \"after\" table (image_features_after.csv) that can be
    compared to the original measurements to see how much the adjustments
    reduced differences between images / conditions.
    """

    if not os.path.isdir(ADJUSTED_IMAGE_DIR):
        print(f"\nNo adjusted image folder found at: {ADJUSTED_IMAGE_DIR}")
        return

    df_after = process_all_images(ADJUSTED_IMAGE_DIR)
    if df_after.empty:
        print("\nNo adjusted images were processed, skipping after-measurement.")
        return

    df_after.to_csv(CSV_AFTER_OUTPUT_PATH, index=False)
    print(f"\nSaved adjusted-image features to: {CSV_AFTER_OUTPUT_PATH}")

    # Quick summary for adjustable features after adjustment
    print("\nFeature summary AFTER adjustment (adjusted images):")
    print_feature_summary(df_after)


def main() -> None:
    print("Image feature extraction (all 17 features)")
    print(f"Looking for images in: {INPUT_DIR}")

    ensure_output_folder_exists()

    # Measure features and run MAD on the original images.
    # This tells us which images are problematic in their native form.
    df = process_all_images(INPUT_DIR)

    if df.empty:
        print("No data to save (no images were found or processed).")
        return

    # Save features to CSV
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to: {CSV_OUTPUT_PATH}")

    # Quick numeric summary for adjustable technical features
    print_feature_summary(df)

    # Outlier flagging
    print("\n Outlier detection")
    outlier_df = flag_mad_outliers(df)
    outlier_df.to_csv(OUTLIER_REPORT_PATH, index=False)
    print(f"Outlier report saved to: {OUTLIER_REPORT_PATH}")

    # Summary: total outliers (all features, all images)
    n_outliers = outlier_df["is_outlier"].sum()
    print(f"Total outlier values: {n_outliers}")

    # Per-image outlier summary to help decide later which images to keep/exclude
    summarize_outliers_by_image(outlier_df)

    # Build a simple adjustment plan for adjustable features based on MAD outliers.
    plan_df = build_adjustment_plan(df, outlier_df)

    # Reset adjusted images folder so each run starts from the original images.
    reset_adjusted_folder()

    # Apply exposure adjustments first (brightness).
    adjust_exposure_from_plan(
        plan_df=plan_df,
        input_dir=INPUT_DIR,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    # Then apply contrast adjustments, stacking on top of exposure changes
    # (if any). Both adjustments are saved into the same adjusted_images folder.
    adjust_contrast_from_plan(
        plan_df=plan_df,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    # Finally apply saturation adjustments (avg_sat), also stacking on top of
    # any previous changes to create one final adjusted image per filename.
    adjust_saturation_from_plan(
        plan_df=plan_df,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    # After all adjustments, re-measure features on the adjusted images
    # so we have an \"after\" table for numeric comparison.
    measure_adjusted_images()


if __name__ == "__main__":
    # This makes sure main() runs only when you run this file directly,
    # not when you import it from somewhere else.
    main()

