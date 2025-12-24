import numpy as np
from pathlib import Path

def inspect():
    try:
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("❌ data/processed/y_train.npy not found. Run prepare_data.py first.")
        return

    print("="*30)
    print("DATA INSPECTION REPORT")
    print("="*30)
    
    print(f"Total Training Labels: {len(y_train)}")
    print(f"Total Test Labels:     {len(y_test)}")
    
    # 1. Check Range
    print(f"\nLabel Range (Training):")
    print(f"   Min: {y_train.min()}")
    print(f"   Max: {y_train.max()}")
    print(f"   Mean: {y_train.mean():.2f}")
    
    # 2. Check for "Crazy" values
    if y_train.max() > 20:
        print("\n⚠️  WARNING: Max label is > 20. Did we extract the ID instead of the Day?")
    else:
        print("\n✅ Labels look like realistic day counts.")

    # 3. Show Samples
    print("\nSample Labels (First 10):")
    print(y_train[:10])

if __name__ == "__main__":
    inspect()