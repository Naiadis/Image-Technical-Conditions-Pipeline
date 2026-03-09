"""
Image standardisation pipeline.

Measures 17 image features, detects outliers within each condition,
finds the best-matching triplet, and adjusts it toward a shared mean.

See README.md for full documentation, references, and folder structure.
"""

import os
import hashlib
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

# Names of the three experimental conditions, used when images live
# in subfolders like data/input/isolated, data/input/in_use, etc.
CONDITION_NAMES = ("isolated", "in_use", "environmental")

# Feature separation for later transformations
# These lists mark which measured features we plan to actively adjust
# (for example via brightness/contrast/saturation/resize) and which ones
# we will mostly monitor and, if needed, use for excluding images.

ADJUSTABLE_FEATURES = [
    "exposure",
    "avg_intensity",
    "contrast",
    "avg_sat",
    "sharpness",
    "height",
    "width",
]

MEASURE_ONLY_FEATURES = [
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

# How many measure-only outliers an image is allowed before we exclude it.
# Images with MORE than this number won't be adjusted (they should be
# replaced with better images or dropped from the experiment).
MAX_MEASURE_ONLY_OUTLIERS = 0

# How far toward the target we move in a single adjustment (0.0 = no change,
# 1.0 = move all the way).  0.5 means we close half the gap, which avoids
# drastic changes that make images look unnatural.
DAMPING = 0.8


def _find_original_image(filename: str) -> str:
    """
    Find the full path of an image by filename, searching recursively
    under INPUT_DIR.  Images live in condition subfolders like
    data/input/isolated/, so we can't just do INPUT_DIR + filename.
    Returns the first match, or an empty string if not found.
    """
    for root, _, files in os.walk(INPUT_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return ""


def run_quality_checks(input_dir: str) -> None:
    """
    Run basic quality checks on all images before processing.
    Catches problems early so you don't waste time on bad data.

    Checks:
      1. Corrupt files   — images that can't be opened by OpenCV.
      2. Duplicate files  — different filenames but identical pixel content
                           (detected via MD5 hash of the file).
      3. Colour mode      — grayscale images that should be RGB/BGR.
      4. Minimum size     — images smaller than 500x500 (too small for
                           reliable feature extraction).
    """

    print("\nRunning quality checks...")
    image_paths = list_image_files(input_dir)

    if not image_paths:
        print("  No images found to check.")
        return

    corrupt = []
    grayscale_imgs = []
    too_small = []
    hash_map: Dict[str, str] = {}
    duplicates = []

    for path in image_paths:
        filename = os.path.basename(path)

        # 1. Corrupt file check: can OpenCV read it?
        img = cv2.imread(path)
        if img is None:
            corrupt.append(filename)
            continue

        # 2. Duplicate check: MD5 hash of the raw file bytes.
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in hash_map:
            duplicates.append((filename, hash_map[file_hash]))
        else:
            hash_map[file_hash] = filename

        # 3. Colour mode: is the image grayscale (single channel)?
        if len(img.shape) < 3 or img.shape[2] == 1:
            grayscale_imgs.append(filename)

        # 4. Minimum resolution.
        h, w = img.shape[:2]
        if h < 500 or w < 500:
            too_small.append((filename, w, h))

    # Report results.
    problems_found = False

    if corrupt:
        problems_found = True
        print(f"\n  CORRUPT files ({len(corrupt)}) — cannot be read:")
        for f in corrupt:
            print(f"    - {f}")

    if duplicates:
        problems_found = True
        print(f"\n  DUPLICATE files ({len(duplicates)}) — identical content:")
        for dup, original in duplicates:
            print(f"    - {dup}  is a copy of  {original}")

    if grayscale_imgs:
        problems_found = True
        print(f"\n  GRAYSCALE images ({len(grayscale_imgs)}) — not RGB/colour:")
        for f in grayscale_imgs:
            print(f"    - {f}")

    if too_small:
        problems_found = True
        print(f"\n  TOO SMALL ({len(too_small)}) — below 500x500 minimum:")
        for f, w, h in too_small:
            print(f"    - {f}: {w}x{h}")

    if not problems_found:
        print(f"  All {len(image_paths)} images passed quality checks.")
    else:
        print("\n  Review the issues above before continuing.")


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
    """
    Find .jpg, .jpeg, .png files under input_dir (recursively).

    This lets us organise images in subfolders such as:
      data/input/isolated, data/input/in_use, data/input/environmental
    without changing the rest of the code.
    """
    if not os.path.isdir(input_dir):
        return []
    allowed = {".jpg", ".jpeg", ".png"}
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name.lower())[1]
            if ext in allowed:
                paths.append(os.path.join(root, name))
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

    # Condition label inferred from folder name, if possible.
    # We look for .../isolated/ or .../in_use/ or .../environmental/ in the path.
    condition = "unknown"
    for cand in CONDITION_NAMES:
        marker = os.sep + cand + os.sep
        if marker in image_path:
            condition = cand
            break


    # Feature dictionary
    features: Dict[str, Any] = {
        "filename": os.path.basename(image_path),
        "condition": condition,
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
    """
    Flag images whose feature values fall outside median ± 2.5 × MAD.

    If images have a 'condition' column with real condition labels,
    MAD is computed WITHIN each condition separately (so each image
    is compared only to others in the same condition).
    Otherwise, MAD is computed globally across all images.
    """

    numeric_cols = [c for c in df.columns if c not in ("filename", "condition")]

    # Check if we have real condition labels (not just "unknown").
    has_conditions = (
        "condition" in df.columns
        and df["condition"].nunique() > 1
        and not (df["condition"] == "unknown").all()
    )

    rows = []

    if has_conditions:
        # Within-condition MAD: each condition is treated independently.
        print("  (Running MAD within each condition separately.)")
        for condition, group in df.groupby("condition"):
            for feat in numeric_cols:
                vals = group[feat].values
                median = np.median(vals)
                mad = _compute_mad(vals)
                k = 2.5
                lower = median - k * mad
                upper = median + k * mad
                for i, fn in enumerate(group["filename"]):
                    v = vals[i]
                    out = v < lower or v > upper
                    rows.append({
                        "filename": fn,
                        "condition": condition,
                        "feature": feat,
                        "value": v,
                        "median": median,
                        "mad": mad,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "is_outlier": out,
                    })
    else:
        # Fallback: global MAD across all images (no condition info).
        print("  (Running MAD globally – no condition subfolders detected.)")
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
                    "condition": "unknown",
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


def print_condition_summary(df: pd.DataFrame, title: str) -> None:
    """
    Print mean and standard deviation for adjustable features per condition.

    This will be most useful once images are organised into folders like:
      data/input/isolated, data/input/in_use, data/input/environmental
    so that the 'condition' column has meaningful values.
    """

    if "condition" not in df.columns:
        print("\nNo 'condition' column found in DataFrame.")
        return

    print(f"\n{title}")
    grouped = df.groupby("condition")
    for condition, group in grouped:
        print(f"\nCondition: {condition} (n={len(group)})")
        for feat in ADJUSTABLE_FEATURES:
            if feat in group.columns:
                mean_val = group[feat].mean()
                std_val = group[feat].std()
                print(f"  {feat:>12}: mean = {mean_val:.2f}, std = {std_val:.2f}")


def summarize_outliers_by_image(
    outlier_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise how many outliers each image has, split by feature type.
    Shows ALL images (including those with zero outliers) so you can
    see the full picture.

    Images with more than MAX_MEASURE_ONLY_OUTLIERS measure-only outliers
    are marked as "EXCLUDE".
    """

    def _feature_type(feat: str) -> str:
        if feat in ADJUSTABLE_FEATURES:
            return "adjustable"
        if feat in MEASURE_ONLY_FEATURES:
            return "measure_only"
        return "other"

    outlier_df = outlier_df.copy()
    outlier_df["feature_type"] = outlier_df["feature"].apply(_feature_type)

    out_only = outlier_df[outlier_df["is_outlier"] == True]

    # Build a base list of ALL images (with condition) from features_df,
    # so images with zero outliers still appear.
    has_cond = "condition" in features_df.columns
    if has_cond:
        all_images = features_df[["filename", "condition"]].drop_duplicates()
    else:
        all_images = features_df[["filename"]].drop_duplicates()
        all_images["condition"] = "unknown"

    # Count outliers per image per feature type.
    if not out_only.empty:
        group_cols = ["condition", "filename"] if "condition" in out_only.columns else ["filename"]
        counts = (
            out_only
            .groupby(group_cols + ["feature_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if "condition" not in counts.columns:
            counts["condition"] = "unknown"
    else:
        counts = pd.DataFrame(columns=["condition", "filename"])

    for col in ["adjustable", "measure_only", "other"]:
        if col not in counts.columns:
            counts[col] = 0

    # Merge with all_images so zero-outlier images appear too.
    summary = all_images.merge(
        counts, on=["condition", "filename"], how="left"
    ).fillna(0)

    for col in ["adjustable", "measure_only", "other"]:
        summary[col] = summary[col].astype(int)

    summary["total_outliers"] = (
        summary["adjustable"] + summary["measure_only"] + summary["other"]
    )

    # Mark images that should be excluded.
    summary["exclude"] = summary["measure_only"] > MAX_MEASURE_ONLY_OUTLIERS

    # Print grouped by condition.
    print(f"\nOutlier summary per image (within-condition, "
          f"exclude if measure_only > {MAX_MEASURE_ONLY_OUTLIERS}):")
    for condition in sorted(summary["condition"].unique()):
        cond_group = summary[summary["condition"] == condition]
        print(f"\n  Condition: {condition}")
        for _, row in cond_group.sort_values("filename").iterrows():
            tag = " << EXCLUDE" if row["exclude"] else ""
            print(
                f"    {row['filename']}: "
                f"adjustable={row['adjustable']}, "
                f"measure_only={row['measure_only']}, "
                f"total={row['total_outliers']}{tag}"
            )

    n_excluded = summary["exclude"].sum()
    n_kept = len(summary) - n_excluded
    print(f"\n  Images to keep: {n_kept}  |  Images to exclude: {n_excluded}")

    return summary


def build_triplet_adjustment_plan(
    features_df: pd.DataFrame,
    triplet_files: List[str],
) -> pd.DataFrame:
    """
    Build an adjustment plan for ONLY the 3 chosen triplet images.

    The target for each adjustable feature is the mean of those 3 images
    (the triplet mean), damped by DAMPING.  This way the 3 images are
    pulled toward each other, not toward some unrelated global average.
    """

    # Keep only the triplet rows from the features table.
    triplet_df = features_df[features_df["filename"].isin(triplet_files)]

    if triplet_df.empty:
        print("\nNo triplet images found in features table.")
        return pd.DataFrame()

    # Triplet mean = the target the 3 images should converge toward.
    triplet_means: Dict[str, float] = {}
    for feat in ADJUSTABLE_FEATURES:
        if feat in triplet_df.columns:
            triplet_means[feat] = triplet_df[feat].mean()

    # For each image × adjustable feature, check if it differs enough
    # from the triplet mean to warrant adjustment.  We use a simple
    # threshold: if the value is more than 5% away from the triplet
    # mean, we plan an adjustment.
    rows = []
    for _, img_row in triplet_df.iterrows():
        fn = img_row["filename"]
        cond = img_row.get("condition", "unknown")
        for feat in ADJUSTABLE_FEATURES:
            if feat not in img_row.index or feat not in triplet_means:
                continue
            current = float(img_row[feat])
            target = triplet_means[feat]
            if target == 0:
                continue
            pct_diff = abs(current - target) / abs(target)
            if pct_diff > 0.05:
                damped = current + DAMPING * (target - current)
                rows.append({
                    "filename": fn,
                    "condition": cond,
                    "feature": feat,
                    "value": current,
                    "target_mean": target,
                    "damped_target": damped,
                })

    if not rows:
        print("\nTriplet images are already very similar — no adjustment needed!")
        return pd.DataFrame()

    plan = pd.DataFrame(rows)

    print(f"\nAdjustment plan for chosen triplet (damping={DAMPING}):")
    for _, row in plan.iterrows():
        print(
            f"  [{row['condition']}] {row['filename']} -> {row['feature']}: "
            f"current={row['value']:.2f}, "
            f"triplet_mean={row['target_mean']:.2f}, "
            f"damped_target={row['damped_target']:.2f}"
        )

    return plan


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
        target_val = float(row["damped_target"])

        img_path = _find_original_image(filename)
        if not img_path:
            print(f"WARNING: Could not find original image: {filename}")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Could not read image for exposure adjustment: {img_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if current_val <= 0:
            print(f"WARNING: Non-positive current exposure for {filename}, skipping.")
            adjusted_v = v
        else:
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

    # Fallback: search recursively under INPUT_DIR (images are in subfolders).
    original_path = _find_original_image(filename)
    if original_path:
        return cv2.imread(original_path)
    return None


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
        target_contrast = float(row["damped_target"])

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
        target_sat = float(row["damped_target"])

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


def adjust_sharpness_from_plan(
    plan_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Adjust sharpness by applying a Gaussian blur to images that are
    too sharp compared to the triplet mean.

    - Only reduces sharpness (blur), never increases it, because
      adding artificial sharpness creates visible artifacts.
    - The blur kernel size controls how much sharpness is reduced.
      A larger kernel = more blur = lower sharpness.
    """

    if plan_df is None or plan_df.empty:
        print("\nNo adjustment plan available, skipping sharpness adjustment.")
        return

    sharp_rows = plan_df[plan_df["feature"] == "sharpness"]
    if sharp_rows.empty:
        print("\nNo sharpness differences in the adjustment plan, nothing to change.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAdjusting sharpness for {len(sharp_rows)} images.")

    for _, row in sharp_rows.iterrows():
        filename = row["filename"]
        current_sharp = float(row["value"])
        target_sharp = float(row["damped_target"])

        # Only blur (reduce sharpness). Increasing sharpness artificially
        # creates visible edge artifacts, so we skip those cases.
        if target_sharp >= current_sharp:
            print(f"  {filename}: already at or below target, skipping.")
            continue

        image = _load_image_preferring_adjusted(filename)
        if image is None:
            print(f"WARNING: Could not read image for sharpness adjustment: {filename}")
            continue

        # We iteratively apply small blurs and check the Laplacian variance
        # until we get close to the target.  This is safer than guessing a
        # single kernel size, and keeps the blur as gentle as possible.
        best_image = image
        for ksize in [3, 5, 7, 9, 11, 13, 15]:
            blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            new_sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            best_image = blurred
            if new_sharp <= target_sharp:
                break

        save_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(save_path, best_image)
        if not success:
            print(f"WARNING: Could not save sharpness-adjusted image to: {save_path}")
        else:
            print(f"  {filename}: sharpness {current_sharp:.1f} -> ~{new_sharp:.1f} "
                  f"(target {target_sharp:.1f}, kernel {ksize}x{ksize})")


def measure_adjusted_images(original_df: pd.DataFrame) -> None:
    """
    Re-measure all features on the adjusted images and save to a new CSV.

    The adjusted images live flat in ADJUSTED_IMAGE_DIR (no subfolders),
    so we carry over the condition label from the original measurements.
    """

    if not os.path.isdir(ADJUSTED_IMAGE_DIR):
        print(f"\nNo adjusted image folder found at: {ADJUSTED_IMAGE_DIR}")
        return

    df_after = process_all_images(ADJUSTED_IMAGE_DIR)
    if df_after.empty:
        print("\nNo adjusted images were processed, skipping after-measurement.")
        return

    # Carry over condition labels from the original features table,
    # because the adjusted images are saved flat (no condition subfolders).
    if "condition" in original_df.columns:
        cond_map = original_df.set_index("filename")["condition"].to_dict()
        df_after["condition"] = df_after["filename"].map(cond_map).fillna("unknown")

    df_after.to_csv(CSV_AFTER_OUTPUT_PATH, index=False)
    print(f"\nSaved adjusted-image features to: {CSV_AFTER_OUTPUT_PATH}")

    print("\nFeature summary AFTER adjustment (adjusted images):")
    print_feature_summary(df_after)

    print_condition_summary(
        df_after,
        title="Per-condition summary AFTER adjustment (adjusted images)",
    )


def find_best_triplets(
    features_df: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    outlier_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find the best-matching triplet of images (one per condition) whose
    technical features are already the most similar.

    For each recommended triplet it also shows:
      - The raw feature values side by side.
      - Which features are flagged as outliers (adjustable or measure-only).
      - A verdict: "ready" / "needs small adjustment" / "has measure-only issues".
    """

    if "condition" not in features_df.columns:
        print("\nCannot find best triplet: no condition labels.")
        return pd.DataFrame()

    # Remove excluded images.
    keep_files = set(features_df["filename"])
    if exclusion_summary is not None and not exclusion_summary.empty:
        excluded = set(
            exclusion_summary.loc[
                exclusion_summary["exclude"] == True, "filename"
            ]
        )
        keep_files -= excluded

    df = features_df[features_df["filename"].isin(keep_files)].copy()

    conditions = sorted(df["condition"].unique())
    if len(conditions) < 3:
        print(f"\nNeed 3 conditions to form a triplet, found {len(conditions)}.")
        return pd.DataFrame()

    iso = df[df["condition"] == "isolated"]
    use = df[df["condition"] == "in_use"]
    env = df[df["condition"] == "environmental"]

    if iso.empty or use.empty or env.empty:
        print("\nAt least one condition has no (non-excluded) images.")
        return pd.DataFrame()

    numeric_cols = [
        c for c in df.columns if c not in ("filename", "condition")
    ]

    # Z-score so all features contribute equally.
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std().replace(0, 1)
    df_z = df.copy()
    df_z[numeric_cols] = (df[numeric_cols] - means) / stds

    iso_z = df_z[df_z["condition"] == "isolated"]
    use_z = df_z[df_z["condition"] == "in_use"]
    env_z = df_z[df_z["condition"] == "environmental"]

    results = []
    for _, r_iso in iso_z.iterrows():
        for _, r_use in use_z.iterrows():
            for _, r_env in env_z.iterrows():
                vals = np.array([
                    [r_iso[f] for f in numeric_cols],
                    [r_use[f] for f in numeric_cols],
                    [r_env[f] for f in numeric_cols],
                ])
                score = float(np.sum(np.ptp(vals, axis=0)))
                results.append({
                    "isolated": r_iso["filename"],
                    "in_use": r_use["filename"],
                    "environmental": r_env["filename"],
                    "score": score,
                })

    if not results:
        print("\nNo valid triplets found.")
        return pd.DataFrame()

    ranking = pd.DataFrame(results)

    # Pre-build a quick lookup: filename -> list of outlier feature names,
    # split into adjustable vs measure-only.
    adj_outliers_by_file: Dict[str, list] = {}
    mo_outliers_by_file: Dict[str, list] = {}
    if outlier_df is not None and not outlier_df.empty:
        flagged = outlier_df[outlier_df["is_outlier"] == True]
        for _, orow in flagged.iterrows():
            fn = orow["filename"]
            feat = orow["feature"]
            if feat in ADJUSTABLE_FEATURES:
                adj_outliers_by_file.setdefault(fn, []).append(feat)
            elif feat in MEASURE_ONLY_FEATURES:
                mo_outliers_by_file.setdefault(fn, []).append(feat)

    # Compute total outliers per triplet so we can sort PERFECT first.
    def _triplet_outlier_counts(row):
        files = [row["isolated"], row["in_use"], row["environmental"]]
        adj = sum(len(adj_outliers_by_file.get(f, [])) for f in files)
        mo = sum(len(mo_outliers_by_file.get(f, [])) for f in files)
        return adj + mo

    ranking["total_outliers"] = ranking.apply(_triplet_outlier_counts, axis=1)

    # Sort: PERFECT (0 outliers) first, then by score within each group.
    ranking = ranking.sort_values(
        ["total_outliers", "score"]
    ).reset_index(drop=True)

    # All features for the comparison table.
    all_features = ADJUSTABLE_FEATURES + MEASURE_ONLY_FEATURES

    print(f"\n{'=' * 60}")
    print(f"Top {min(top_n, len(ranking))} best-matching triplets")
    print(f"(PERFECT first, then sorted by similarity score)")
    print(f"{'=' * 60}")

    for i, row in ranking.head(top_n).iterrows():
        triplet_files = [row["isolated"], row["in_use"], row["environmental"]]

        all_adj = sum(len(adj_outliers_by_file.get(f, [])) for f in triplet_files)
        all_mo = sum(len(mo_outliers_by_file.get(f, [])) for f in triplet_files)
        if all_adj == 0 and all_mo == 0:
            verdict = "PERFECT"
        elif all_mo == 0:
            verdict = f"GOOD ({all_adj} adjustable outlier(s))"
        else:
            verdict = f"OK ({all_adj} adj + {all_mo} measure-only)"

        print(f"\n  #{i + 1}  score = {row['score']:.2f}  [{verdict}]")

        triplet_df = features_df[features_df["filename"].isin(triplet_files)]

        cond_vals: Dict[str, Dict[str, float]] = {}
        for _, img_row in triplet_df.iterrows():
            cond_vals[img_row["condition"]] = {
                f: img_row[f] for f in numeric_cols
            }

        for cond_key, file_key in [
            ("isolated", "isolated"),
            ("in_use", "in_use"),
            ("environmental", "environmental"),
        ]:
            fn = row[file_key]
            adj_out = adj_outliers_by_file.get(fn, [])
            mo_out = mo_outliers_by_file.get(fn, [])

            if not adj_out and not mo_out:
                status = "no outliers"
            elif adj_out and not mo_out:
                status = f"adjustable: {', '.join(adj_out)}"
            elif mo_out and not adj_out:
                status = f"measure-only: {', '.join(mo_out)}"
            else:
                status = (
                    f"adjustable: {', '.join(adj_out)}; "
                    f"measure-only: {', '.join(mo_out)}"
                )
            print(f"    {cond_key + ':':16s} {fn}")
            print(f"      -> {status}")

        # Full feature comparison table (all features).
        print(f"    {'feature':>14}  {'isolated':>10}  {'in_use':>10}  {'environ.':>10}")
        for feat in all_features:
            v_iso = cond_vals.get("isolated", {}).get(feat, 0)
            v_use = cond_vals.get("in_use", {}).get(feat, 0)
            v_env = cond_vals.get("environmental", {}).get(feat, 0)
            print(f"    {feat:>14}  {v_iso:>10.1f}  {v_use:>10.1f}  {v_env:>10.1f}")

    return ranking


def main() -> None:
    print("=" * 60)
    print("Image standardisation pipeline")
    print("=" * 60)
    print(f"Looking for images in: {INPUT_DIR}")

    ensure_output_folder_exists()

    # ------------------------------------------------------------------
    # STEP 0  Quality checks (corrupt, duplicates, grayscale, too small).
    # ------------------------------------------------------------------
    run_quality_checks(INPUT_DIR)

    # ------------------------------------------------------------------
    # STEP 1  Measure all 17 features on the original images.
    # ------------------------------------------------------------------
    print("\n--- Step 1: Measure features on original images ---")
    df = process_all_images(INPUT_DIR)

    if df.empty:
        print("No data to save (no images were found or processed).")
        return

    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to: {CSV_OUTPUT_PATH}")

    print_feature_summary(df)

    print_condition_summary(
        df,
        title="Per-condition summary BEFORE adjustment (original images)",
    )

    # ------------------------------------------------------------------
    # STEP 2  Within-condition MAD outlier detection.
    #         Each image is compared to others in its own condition.
    #         This helps decide which images to EXCLUDE (measure-only
    #         outliers) and which to ADJUST (adjustable-feature outliers).
    # ------------------------------------------------------------------
    print("\n--- Step 2: Within-condition outlier detection (MAD) ---")
    outlier_df = flag_mad_outliers(df)
    outlier_df.to_csv(OUTLIER_REPORT_PATH, index=False)
    print(f"Outlier report saved to: {OUTLIER_REPORT_PATH}")

    n_outliers = outlier_df["is_outlier"].sum()
    print(f"Total outlier flags: {n_outliers}")

    exclusion_summary = summarize_outliers_by_image(outlier_df, df)

    # ------------------------------------------------------------------
    # STEP 3  Find the best-matching triplet (1 per condition) that
    #         already has the most similar technical values.
    # ------------------------------------------------------------------
    print("\n--- Step 3: Find best triplet ---")
    ranking = find_best_triplets(df, exclusion_summary, outlier_df, top_n=5)

    if ranking is None or ranking.empty:
        print("\nCould not find any valid triplets. Pipeline stops here.")
        print("=" * 60)
        return

    # Pick the #1 best triplet automatically.
    best = ranking.iloc[0]
    triplet_files = [best["isolated"], best["in_use"], best["environmental"]]
    print(f"\nChosen triplet (best match):")
    print(f"  isolated:      {best['isolated']}")
    print(f"  in_use:        {best['in_use']}")
    print(f"  environmental: {best['environmental']}")

    # ------------------------------------------------------------------
    # STEP 4  Adjust ONLY the chosen triplet toward their shared mean.
    #         This pulls the 3 images closer to each other, not toward
    #         some unrelated global average.
    # ------------------------------------------------------------------
    print("\n--- Step 4: Adjust triplet toward shared mean ---")
    plan_df = build_triplet_adjustment_plan(df, triplet_files)

    reset_adjusted_folder()

    adjust_exposure_from_plan(
        plan_df=plan_df,
        input_dir=INPUT_DIR,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    adjust_contrast_from_plan(
        plan_df=plan_df,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    adjust_saturation_from_plan(
        plan_df=plan_df,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    adjust_sharpness_from_plan(
        plan_df=plan_df,
        output_dir=ADJUSTED_IMAGE_DIR,
    )

    # ------------------------------------------------------------------
    # STEP 5  Re-measure the adjusted triplet images ("after").
    # ------------------------------------------------------------------
    print("\n--- Step 5: Re-measure adjusted triplet ---")
    measure_adjusted_images(df)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    # This makes sure main() runs only when you run this file directly,
    # not when you import it from somewhere else.
    main()

