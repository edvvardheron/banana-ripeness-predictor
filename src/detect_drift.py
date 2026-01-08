import numpy as np
import tensorflow as tf
import os
import glob
from pathlib import Path
from PIL import Image
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = "models/banana_mobilenet.keras"
REFERENCE_DATA_PATH = "data/processed/X_train.npy"
NEW_DATA_FOLDER = "data/incoming"  # Put new raw images here to test
DRIFT_THRESHOLD = 0.1  # Sensitivity (Lower = more sensitive)

def load_new_images(folder_path):
    """Loads raw images from a folder and processes them for the model."""
    images = []
    valid_exts = {".jpg", ".jpeg", ".png"}
    
    path = Path(folder_path)
    if not path.exists():
        print(f"Folder {folder_path} does not exist. Create it and add images.")
        return np.array([])

    for file_path in path.glob("*"):
        if file_path.suffix.lower() not in valid_exts:
            continue
        
        try:
            # Resize to match model input (299x299)
            img = Image.open(file_path).convert("RGB")
            img = img.resize((299, 299))
            img_array = np.array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            
    return np.array(images)

def get_feature_extractor(full_model):
    """
    Strips off the final 'prediction' layer.
    Returns the 'embedding' layer (the high-level features).
    This works independently of any 'Gatekeeper' logic.
    """
    # 1. Try finding the Pooling layer by name (standard Keras naming)
    layer_name = 'global_average_pooling2d'
    try:
        embedding_output = full_model.get_layer(layer_name).output
        return tf.keras.Model(inputs=full_model.input, outputs=embedding_output)
    except ValueError:
        pass # Continue to search strategy

    # 2. Robust Fallback: Search by Layer Type string
    # This helps if the layer is named 'global_average_pooling2d_1' etc.
    for layer in full_model.layers:
        if "GlobalAveragePooling2D" in layer.__class__.__name__:
             print(f"Found embedding layer by type: {layer.name}")
             return tf.keras.Model(inputs=full_model.input, outputs=layer.output)

    # 3. Final Fallback: Grab the 3rd to last layer (Usually the Dense layer before Dropout)
    # Architecture: [ ... GlobalAvgPool -> Dense(128) -> Dropout -> Output ]
    # -1 is Output, -2 is Dropout, -3 is Dense(128). 
    # The Dense(128) layer is actually a great place to check for drift.
    print(f"Specific pooling layer not found. Using hidden layer: {full_model.layers[-3].name}")
    return tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-3].output)

def detect():
    print("Loading Resources...")
    
    # 1. Load Reference Data (Training Set)
    if not os.path.exists(REFERENCE_DATA_PATH):
        print("Processed training data not found. Run prepare_data.py first.")
        return
    X_ref = np.load(REFERENCE_DATA_PATH)
    
    # 2. Load New Data (Production/Incoming)
    X_new = load_new_images(NEW_DATA_FOLDER)
    
    if len(X_new) == 0:
        print("No new images found to check.")
        return

    # 3. Load Model & Create Feature Extractor
    full_model = tf.keras.models.load_model(MODEL_PATH)
    feature_model = get_feature_extractor(full_model)

    print("Extracting Embeddings (this may take a moment)...")
    # Get the "Brain Activity" for old vs new data
    ref_embeddings = feature_model.predict(X_ref, verbose=0)
    new_embeddings = feature_model.predict(X_new, verbose=0)

    # 4. Calculate Drift (Wasserstein Distance)
    # We look at the distribution of the embeddings.
    # We take the mean across features to get a simple scalar "complexity score" per image
    ref_mean = np.mean(ref_embeddings, axis=1)
    new_mean = np.mean(new_embeddings, axis=1)

    drift_score = wasserstein_distance(ref_mean, new_mean)
    
    print("\n" + "="*30)
    print("DRIFT DETECTION REPORT")
    print("="*30)
    print(f"Reference Images: {len(X_ref)}")
    print(f"New Images:       {len(X_new)}")
    print(f"Drift Score:      {drift_score:.4f}")
    print(f"Threshold:        {DRIFT_THRESHOLD}")
    
    if drift_score > DRIFT_THRESHOLD:
        print("\nDRIFT DETECTED!")
        print("The new data looks significantly different from the training data.")
        print("Recommendation: Label these new images and Retrain the model.")
    else:
        print("\nData looks stable.")
        print("The new images follow a similar distribution to the training set.")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(ref_mean, bins=30, alpha=0.5, label='Training Data', density=True, color='blue')
    plt.hist(new_mean, bins=30, alpha=0.5, label='New Data', density=True, color='red')
    plt.title(f"Data Distribution Shift (Score: {drift_score:.4f})")
    plt.xlabel("Image Feature Complexity")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("drift_report.png")
    print("\nSaved visualization to 'drift_report.png'")

if __name__ == "__main__":
    detect()