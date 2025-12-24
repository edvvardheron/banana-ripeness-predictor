import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# --- Config ---
MODEL_PATH = "models/banana_model.keras"
DATA_DIR = "data/processed"

def diagnose():
    # 1. Load Data and Model
    print(f"📂 Loading data from {DATA_DIR}...")
    X_test = np.load(Path(DATA_DIR) / "X_test.npy")
    y_test = np.load(Path(DATA_DIR) / "y_test.npy")
    
    print(f"🧠 Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # 2. Run Predictions
    print("🔮 Running predictions...")
    predictions = model.predict(X_test, verbose=0).flatten()

    # 3. Calculate Errors
    errors = np.abs(predictions - y_test)
    
    # 4. Find the Worst Mistakes
    # Get indices of the sorted errors (largest to smallest)
    worst_indices = np.argsort(errors)[::-1]
    
    print("\n" + "="*40)
    print("🛑 TOP 5 WORST PREDICTIONS")
    print("="*40)
    
    for i in range(5):
        if i >= len(worst_indices): break
        
        idx = worst_indices[i]
        actual = y_test[idx]
        pred = predictions[idx]
        error = errors[idx]
        
        print(f"Rank #{i+1}:")
        print(f"   Actual Days:    {actual}")
        print(f"   Predicted Days: {pred:.2f}")
        print(f"   Error:          {error:.2f} days off")
        print("-" * 20)

    # 5. Overall Stats
    mse = np.mean(errors ** 2)
    mae = np.mean(errors)
    print(f"\n📊 Overall Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae:.2f}")
    print(f"   Root Mean Sq Error (RMSE): {np.sqrt(mse):.2f}")

    print("\n💡 TIP: If the 'Actual' is consistently 0 but Predicted is high,")
    print("        your model might be confusing green bananas with ripe ones.")

if __name__ == "__main__":
    diagnose()