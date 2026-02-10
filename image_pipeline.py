"""
Simple image measurement script for your thesis.

IMPORTANT: I am keeping this as simple and commented as possible.
You do NOT need to understand everything at once – focus on the comments.

What this script does (first basic version):
1. Looks in a folder for images (jpg, jpeg, png).
2. For each image, it measures:
   - height, width, height+width (composition)
   - average intensity (brightness / exposure)
   - contrast (how spread out the brightness is)
   - sharpness (how clear the edges are)
   - average hue and saturation (basic colour info)
   - colourfulness (how strong the colours are overall)
3. Saves all measurements into a CSV file (a table you can open in Excel / SPSS).

Later, we can extend this with:
   - more features (texture, central HSV metrics, etc.)
   - functions that actually transform images to standardise brightness/contrast/saturation.
"""

import os
from typing import List, Dict

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import pandas as pd  # pandas for table-like data (DataFrame)


# =========================
# BASIC SETTINGS – CHANGE THESE LATER
# =========================

# TODO: change this folder later to where you store your input images.
# For now, you can create a "data/input" folder inside the repo and drop test images there.
INPUT_DIR = "data/input"

# TODO: change this folder later if you want processed images somewhere else.
# In this first version we only SAVE A CSV here, not new images yet.
OUTPUT_DIR = "data/output"

# TODO: change this CSV name later if you want a different file name.
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "image_features_basic.csv")


def ensure_output_folder_exists() -> None:
    """
    Make sure the output folder exists.
    If it does not exist, create it.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_image_files(input_dir: str) -> List[str]:
    """
    Find image files in the given folder.
    We keep this very simple: only look at the TOP LEVEL of the folder,
    and only accept .jpg, .jpeg, .png files.
    """
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_paths: List[str] = []

    if not os.path.isdir(input_dir):
        # The folder does not exist – the user will see a clear message later.
        return file_paths

    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        if not os.path.isfile(full_path):
            # Skip subfolders in this first simple version.
            continue

        _, ext = os.path.splitext(name.lower())
        if ext in allowed_extensions:
            file_paths.append(full_path)

    return file_paths


def compute_basic_features_for_image(image_bgr: np.ndarray, image_path: str) -> Dict[str, float]:
    """
    Compute a basic set of features for ONE image.

    Input:
        image_bgr: the image loaded by OpenCV, in BGR colour format
        image_path: full path to the image (used only for the filename column)

    Output:
        A dictionary with simple numeric features. Each key will become a column in the CSV.
    """
    # Basic size / composition features
    height, width = image_bgr.shape[:2]
    height_plus_width = height + width

    # Convert to grayscale (single channel) for intensity / contrast / sharpness
    # NOTE: OpenCV uses BGR by default (Blue, Green, Red).
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Average intensity (brightness / exposure)
    avg_intensity = float(np.mean(image_gray))

    # Contrast as the standard deviation of intensity
    contrast = float(np.std(image_gray))

    # Sharpness: variance of Laplacian – higher = sharper (more edges)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Convert to HSV for colour features (Hue, Saturation, Value)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(image_hsv)

    # Average hue, saturation, and value
    avg_hue = float(np.mean(h_channel))
    avg_saturation = float(np.mean(s_channel))
    avg_value = float(np.mean(v_channel))

    # Colourfulness metric based on Hasler and Süsstrunk (2003) style formula.
    # This is a standard way to estimate "how colourful" an image is.
    b_channel, g_channel, r_channel = cv2.split(image_bgr.astype("float"))
    rg = np.abs(r_channel - g_channel)
    yb = np.abs(0.5 * (r_channel + g_channel) - b_channel)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    colourfulness = float(
        np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    )

    # Build the feature dictionary.
    # NOTE: Keys here will be the column names in the CSV.
    features: Dict[str, float] = {
        "filename": os.path.basename(image_path),
        "height": float(height),
        "width": float(width),
        "height_plus_width": float(height_plus_width),
        "avg_intensity": avg_intensity,
        "contrast": contrast,
        "sharpness": sharpness,
        "avg_hue": avg_hue,
        "avg_saturation": avg_saturation,
        "avg_value": avg_value,
        "colourfulness": colourfulness,
    }

    return features


def process_all_images(input_dir: str) -> pd.DataFrame:
    """
    Main helper that:
    1. Finds all images in the input folder.
    2. Computes basic features for each image.
    3. Returns a pandas DataFrame with one row per image.
    """
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

        features = compute_basic_features_for_image(image, path)
        all_feature_rows.append(features)

    if not all_feature_rows:
        print("No valid images were processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_feature_rows)
    return df


def main() -> None:
    """
    Entry point of the script.
    This is what runs when you call:  python image_pipeline.py
    """
    print("=== Image feature extraction (basic version) ===")
    print(f"Looking for images in: {INPUT_DIR}")

    ensure_output_folder_exists()

    df = process_all_images(INPUT_DIR)

    if df.empty:
        print("No data to save (no images were found or processed).")
        return

    # Save the results to a CSV file so you can open it in Excel / SPSS.
    df.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"\nDone. Saved {len(df)} rows (one per image) to:")
    print(f"  {CSV_OUTPUT_PATH}")
    print("\nYou can open this CSV file in Excel, SPSS, etc.")


if __name__ == "__main__":
    # This makes sure main() runs only when you run this file directly,
    # not when you import it from somewhere else.
    main()

