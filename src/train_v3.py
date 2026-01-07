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
from sklearn.model_selection import train_test_split

# Config
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

params.update({
    "fine_tune_epochs": 10,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_unfreeze_layers": 50,
})

def load_data(input_dir):
    try:
        input_path = Path(input_dir)
        X_train = np.load(input_path / "X_train.npy")
        X_test = np.load(input_path / "X_test.npy")
        y_train = np.load(input_path / "y_train.npy")
        y_test = np.load(input_path / "y_test.npy")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure to run prepare_data.py first.")
        exit(1)

def build_transfer_model(input_shape):
    """
    Uses MobileNetV2 (pre-trained on ImageNet) as the feature extractor.
    """

    # 1. Input
    inputs = Input(shape=input_shape)

    # 2. Rescaling
    # MobileNetV2 expects pixels between [-1, 1]. 
    # Data is currently [0, 1]. This layer fixes it automatically.
    # (x * 2) -> [0, 2] -> minus 1 -> [-1, 1]
    x = Rescaling(scale=2.0, offset=-1.0)(inputs)

    # 3. The Pre-Trained Model
    # include_top=False (means remove the layer that predicts cats/dogs)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model. We don't want to destroy the pre-learned patterns.
    base_model.trainable = False
    
    # Pass data through the base
    x = base_model(x, training=False)
    
    # 4. Flattening
    # GlobalAveragePooling is smarter than Flatten() for modern networks
    x = GlobalAveragePooling2D()(x)
    
    # 5. Implementing Dropout and Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(params["dropout_rate"])(x)
    
    # Output: Days (ReLU prevents negative days)
    outputs = Dense(1, activation='relu')(x)
    
    model = Model(inputs, outputs)
    # expose the base model for optional fine-tuning by the trainer
    model.base_model = base_model
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train():
    mlflow.set_experiment("Banana_Readiness_Transfer")
    mlflow.tensorflow.autolog

    with mlflow.start_run():
        mlflow.log_params(params)
        
        print("Loading data...")
        X_train, X_test, y_train, y_test = load_data(params["input_dir"])
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=params["seed"])
        
        # Heavy Augmentation to prevent overfitting the pre-trained features
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        print("Downloading MobileNetV2 and building model...")
        model = build_transfer_model(X_train.shape[1:])
        
        model.summary()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        print("Starting Transfer Learning...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Final Test MAE: {mae:.2f} days")
        
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_mae", mae)
        
        Path(params["model_dir"]).mkdir(exist_ok=True)
        model.save(Path(params["model_dir"]) / "banana_mobilenet.keras")

        # --- Fine-tuning stage: unfreeze the last N layers of the base model ---
        try:
            base_model = model.base_model
        except AttributeError:
            print("Base model not exposed on the built model; skipping fine-tuning.")
        else:
            print("Starting fine-tuning: unfreezing base model layers...")
            base_model.trainable = True
            # Freeze all layers except the last N
            unfreeze_n = params.get("fine_tune_unfreeze_layers", 50)
            if unfreeze_n <= 0:
                print("fine_tune_unfreeze_layers <= 0: skipping unfreeze step")
            else:
                n_layers = len(base_model.layers)
                start = max(0, n_layers - unfreeze_n)
                for layer in base_model.layers[:start]:
                    layer.trainable = False

                # Recompile with a lower learning rate for fine-tuning
                optimizer = tf.keras.optimizers.Adam(learning_rate=params.get("fine_tune_learning_rate", 1e-5))
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                # Create fresh callbacks for fine-tuning with fresh state
                ft_early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                ft_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                # Shorter training for fine-tuning
                print(f"Fine-tuning for {params['fine_tune_epochs']} epochs (unfreezing last {unfreeze_n} layers)...")
                ft_history = model.fit(
                    datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
                    validation_data=(X_val, y_val),
                    epochs=params["fine_tune_epochs"],
                    callbacks=[ft_early_stop, ft_reduce_lr],
                    verbose=1
                )

                ft_loss, ft_mae = model.evaluate(X_test, y_test, verbose=0)
                print(f"\n✅ Fine-tuned Test MAE: {ft_mae:.2f} days")
                mlflow.log_metric("fine_tune_test_mae", float(ft_mae))
                mlflow.log_metric("fine_tune_test_loss", float(ft_loss))

                # Save fine-tuned model
                model.save(Path(params["model_dir"]) / "banana_mobilenet_finetuned.keras")

if __name__ == "__main__":
    train()