import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Config ---
params = {
    "seed": 42,
    "input_dir": "data/processed",
    "model_dir": "models",
    "batch_size": 16,        # Smaller batch size for small datasets
    "epochs": 100,           # Give it more time
    "learning_rate": 0.0001, # Slower, more careful learning
    "dropout_rate": 0.5
}

tf.random.set_seed(params["seed"])
np.random.seed(params["seed"])

def load_data(input_dir):
    input_path = Path(input_dir)
    X_train = np.load(input_path / "X_train.npy")
    X_test = np.load(input_path / "X_test.npy")
    y_train = np.load(input_path / "y_train.npy")
    y_test = np.load(input_path / "y_test.npy")
    return X_train, X_test, y_train, y_test

def build_improved_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        
        # Dense Layer
        Dense(64, activation='relu'),
        Dropout(params["dropout_rate"]),
        
        # Output Layer: ReLU prevents negative predictions
        Dense(1, activation='relu') 
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train():
    mlflow.set_experiment("Banana_Readiness_v2")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        print("📂 Loading data...")
        X_train, X_test, y_train, y_test = load_data(params["input_dir"])
        
        # Check shapes
        print(f"   Training on {len(X_train)} images.")

        # Augmentation
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2], # Light variations help a lot!
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        print("🏗️ Building improved model...")
        model = build_improved_model(X_train.shape[1:])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # LR Scheduler: If loss stops dropping, slow down learning rate automatically
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

        print("🚀 Starting training (v2)...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
            validation_data=(X_test, y_test),
            epochs=params["epochs"],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Final Test MAE: {mae:.2f} days")
        
        # Save
        Path(params["model_dir"]).mkdir(exist_ok=True)
        model.save(Path(params["model_dir"]) / "banana_model_v2.keras")

if __name__ == "__main__":
    train()