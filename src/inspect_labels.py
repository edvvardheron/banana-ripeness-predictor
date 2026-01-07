import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect():
    try:
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("data/processed/y_train.npy not found. Run prepare_data.py first.")
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
        print("\nWARNING: Max label is > 20. Extracted the ID instead of the Day?")
    else:
        print("\nLabels look like realistic day counts.")

    # 3. Show Samples
    print("\nSample Labels (First 10):")
    print(y_train[:10])

    # 4. Histogram of Labels
    print("\nGenerating histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(y_train, bins=20, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Training Data Label Distribution')
    axes[0].set_xlabel('Days of Ripeness')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(y_test, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title('Test Data Label Distribution')
    axes[1].set_xlabel('Days of Ripeness')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/label_histogram.png', dpi=100)
    print("Histogram saved to: data/processed/label_histogram.png")
    plt.show()

if __name__ == "__main__":
    inspect()