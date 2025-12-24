import os
import yaml
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
# Again, ideally loaded from params.yaml
params = {
    "seed": 42,
    "input_dir": "data/processed",
    "model_dir": "models",
    "batch_size": 30,
    "epochs": 50,           # We set this high, but Early Stopping will cut it short
    "learning_rate": 0.0001,
    "dropout_rate": 0.5     # REGULARIZATION: 50% of neurons dropped
}

# Ensure reproducibility
tf.random.set_seed(params["seed"])
np.random.seed(params["seed"])

def load_data(input_dir):
    """Loads the numpy arrays saved in Step 2."""
    input_path = Path(input_dir)
    X_train = np.load(input_path / "X_train.npy")
    X_test = np.load(input_path / "X_test.npy")
    y_train = np.load(input_path / "y_train.npy")
    y_test = np.load(input_path / "y_test.npy")
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """
    Defines a simple Convolutional Neural Network (CNN).
    We use a Regression output (1 neuron) to predict 'Days'.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # --- Feature Extraction (The "Eye") ---
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
    
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),  
        
        Flatten(),
        
        # --- Classification/Regression (The "Brain") ---
        Dense(64, activation='relu'),
        
        # REGULARIZATION: Dropout
        # Randomly turns off neurons to prevent memorization
        Dropout(params["dropout_rate"]),
        
        # Output Layer: Single neuron for regression (predicting a number)
        Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    
    # We use Mean Squared Error (MSE) because we are predicting a number (days), not a class.
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae']) # Mean Absolute Error (e.g., "off by 1.2 days")
    return model

def train():
    # 1. Setup MLflow Tracking
    # This creates a local "mlruns" folder to store your experiment history
    mlflow.set_experiment("Banana_Readiness_Prediction")
    mlflow.tensorflow.autolog() # Automatically logs metrics, params, and model artifacts!

    with mlflow.start_run():
        print("📂 Loading data...")
        X_train, X_test, y_train, y_test = load_data(params["input_dir"])
        
        # 2. Data Augmentation (REGULARIZATION)
        # Create "fake" variations of our bananas to expand the dataset
        datagen = ImageDataGenerator(
            rotation_range=20,      # Rotate slightly
            width_shift_range=0.2,  # Shift left/right
            height_shift_range=0.2, # Shift up/down
            #brightness_range=[0.8, 1.2],
            horizontal_flip=True,   # Mirror image
            fill_mode='nearest'
        )
        
        print("🏗️ Building model...")
        model = build_model(X_train.shape[1:])
        model.summary()
        
        # 3. Early Stopping (REGULARIZATION)
        # Stop training if validation loss doesn't improve for 5 epochs
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        print("🚀 Starting training...")
        # Note: We pass datagen.flow() to train on augmented data
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
            validation_data=(X_test, y_test),
            epochs=params["epochs"],
            callbacks=[early_stop],
            verbose=1
        )
        
        # 4. Evaluation
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Final Test MAE: {mae:.2f} days (On average, the prediction is off by {mae:.2f} days)")
        
        # 5. Save Model locally (MLflow also saves it, but good to have a copy)
        Path(params["model_dir"]).mkdir(exist_ok=True)
        model_path = Path(params["model_dir"]) / "banana_model.keras"
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    train()