import os
import sys
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
# In a mature MLOps project, these would be loaded from a params.yaml file
params = {
    "seed": 42,
    "test_size": 0.2,
    "image_size": (224, 224),
    "input_dir": "data/raw",
    "output_dir": "data/processed"
}

def parse_days_from_filename(filename):
    """
    Extracts the target label (days) from the filename.
    Assumes filename format like: 'banana_date_days_X.jpg'
    Logic: Looks for 'days_' and takes the number immediately following.
    """
    name = filename.lower()
    try:
        # Example logic: split by underscores, find the part that is a digit
        # Customize this based on your actual naming convention!
        # If filename is 'banana_2024_days_5.jpg', this splits to parts.
        
        # Simple fallback: Extract the first digit found in the name
        import re
        match = re.search(r'days_(\d+)', name)
        if match:
            return int(match.group(1))
        
        # Fallback 2: Just look for any digit if "days_" isn't there
        match = re.search(r'(\d+)', name)
        if match:
            return int(match.group(1))
            
        raise ValueError("No digit found")
    except Exception as e:
        print(f"Warning: Could not extract label from {filename}: {e}")
        return None

def process_data():
    input_path = Path(params["input_dir"])
    output_path = Path(params["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    print(f"📂 Reading images from {input_path}...")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Iterate over files
    for file_path in input_path.glob("*"):
        if file_path.suffix.lower() not in valid_extensions:
            continue

        # 1. Extract Label
        days = parse_days_from_filename(file_path.name)
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
        print("❌ No valid images found! Check your data directory and filenames.")
        sys.exit(1)

    X = np.array(images)
    y = np.array(labels)

    print(f"✅ Loaded {len(X)} images. Shape: {X.shape}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    # 4. Save to Disk (Compressed NumPy format)
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_test.npy", y_test)

    print(f"💾 Processed data saved to {output_path}")

if __name__ == "__main__":
    process_data()
