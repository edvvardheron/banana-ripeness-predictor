import os
import sys
import re
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
params = {
    "seed": 42,
    "test_size": 0.2,
    "image_size": (299, 299),
    "input_dir": "data/raw",
    "output_dir": "data/processed"
}

def parse_days_from_filename(filename):
    """
    Extracts the target label (days) from the end of the filename.
    Target Format: '10000020308_1_days.jpg' -> Label: 1
    """
    # 1. Remove the extension (.jpg) to get just the name
    # e.g., "10000020308_1_days.jpg" -> "10000020308_1_days"
    stem = Path(filename).stem
    
    # 2. Extract strictly from the end
    # Regex Breakdown:
    # _       : Matches the underscore before the number
    # (\d+)   : Captures the digits (This is the label)
    # _days   : Matches the literal text "_days"
    # $       : Anchors to the END of the string
    match = re.search(r'_(\d+)_days$', stem)
    
    if match:
        return int(match.group(1))
    
    # Debugging: Print warning if format doesn't match
    # This helps catch files that are named incorrectly
    print(f"Warning: Filename '{filename}' does not match format '..._X_days'")
    return None

def process_data():
    input_path = Path(params["input_dir"])
    output_path = Path(params["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    print(f"Reading images from {input_path}...")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Iterate over files
    for file_path in input_path.glob("*"):
        if file_path.suffix.lower() not in valid_extensions:
            continue

        # 1. Extract Label
        days = parse_days_from_filename(file_path.name)
        
        # If days is None, it means the regex didn't match. Skip file.
        if days is None:
            continue

        # 2. Load and Resize Image
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB") # Ensure 3 channels
                img = img.resize(params["image_size"])
                img_array = np.array(img) / 255.0 # Normalize 0-1
                
                images.append(img_array)
                labels.append(days)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not images:
        print("No valid images found! Check your filenames match '..._X_days.jpg'")
        sys.exit(1)

    X = np.array(images)
    y = np.array(labels)

    print(f"Loaded {len(X)} images with correct labels.")
    print(f"Example Label: {y[0]}") 

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    # 4. Save to Disk
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_test.npy", y_test)

    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_data()