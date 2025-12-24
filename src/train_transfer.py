import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Config ---
params = {
    "seed": 42,
    "input_dir": "data/processed",
    "model_dir": "models",
    "batch_size": 16,        
    "epochs": 50,            
    "learning_rate": 0.0001, 
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

def build_transfer_model(input_shape):
    """
    Uses MobileNetV2 (pre-trained on ImageNet) as the feature extractor.
    """
    # 1. Input
    inputs = Input(shape=input_shape)

    # 2. Rescaling
    # MobileNetV2 expects pixels between [-1, 1]. 
    # Your data is currently [0, 1]. This layer fixes it automatically.
    # (x * 2) -> [0, 2] -> minus 1 -> [-1, 1]
    x = Rescaling(scale=2.0, offset=-1.0)(inputs)

    # 3. The Pre-Trained Brain (Base)
    # include_top=False means "Cut off the head" (remove the layer that predicts cats/dogs)
    # weights='imagenet' means "Download the smarts"
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    # FREEZE the base model. We don't want to destroy the pre-learned patterns.
    base_model.trainable = False
    
    # Pass our data through the base
    x = base_model(x, training=False)
    
    # 4. The Bridge (Flattening)
    # GlobalAveragePooling is smarter than Flatten() for modern networks
    x = GlobalAveragePooling2D()(x)
    
    # 5. The Custom Head (Your specific Banana logic)
    x = Dense(128, activation='relu')(x)
    x = Dropout(params["dropout_rate"])(x)
    
    # Output: Days (ReLU prevents negative days)
    outputs = Dense(1, activation='relu')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train():
    mlflow.set_experiment("Banana_Readiness_Transfer")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        print("📂 Loading data...")
        X_train, X_test, y_train, y_test = load_data(params["input_dir"])
        
        # Heavy Augmentation to prevent overfitting the pre-trained brain
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        print("🏗️ Downloading MobileNetV2 and building model...")
        # Note: This will download about 14MB of data the first time run
        model = build_transfer_model(X_train.shape[1:])
        
        model.summary()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        print("🚀 Starting Transfer Learning...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
            validation_data=(X_test, y_test),
            epochs=params["epochs"],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Final Test MAE: {mae:.2f} days")
        
        Path(params["model_dir"]).mkdir(exist_ok=True)
        model.save(Path(params["model_dir"]) / "banana_mobilenet.keras")

if __name__ == "__main__":
    train()