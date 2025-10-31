import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import os
from datetime import datetime

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Task 2.1: Data Loading & Preprocessing
def load_and_prepare_data():
    print("--- Task 2.1: Loading and Preparing Procedural Data ---")
    
    # Load the new procedural training data
    X_train_full = np.load(config.PROCEDURAL_X_TRAIN_PATH)['X_train']
    y_train_full = np.load(config.PROCEDURAL_Y_TRAIN_PATH)
    
    # Load the procedural test data (for saving later, not for training)
    X_test = np.load(config.PROCEDURAL_X_TEST_PATH)['X_test']
    y_test = np.load(config.PROCEDURAL_Y_TEST_PATH)

    print(f"Total training samples: {len(y_train_full)}")
    print(f"Total testing samples: {len(y_test)}")

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    # Load the procedural scaler
    with open(config.PROCEDURAL_SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Normalize the training and validation sets
    X_train = scaler.transform(X_train.reshape(-1, 7)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, 7)).reshape(X_val.shape)

    print(f"  X_train.shape = {X_train.shape}")
    print(f"  y_train.shape = {y_train.shape}")
    print(f"  X_val.shape = {X_val.shape}")
    print(f"  y_val.shape = {y_val.shape}")
    print(f"  X_test.shape = {X_test.shape} (from dedicated test set)")
    print(f"  y_test.shape = {y_test.shape}")
    print("Data preparation complete.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Task 2.2: Define LSTM Architecture
def build_model():
    print("\n--- Task 2.2: Building LSTM Model ---")
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(60, 7)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.LSTM(16, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(13, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    print("Model built successfully.")
    return model

# Task 2.3: Train Model with Callbacks
def run_training(model, X_train, y_train, X_val, y_val):
    print("\n--- Task 2.3: Training Model ---")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_LOG_DIR, histogram_freq=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(config.HEALTH_LSTM_MODEL_PATH)
    
    # Save training history
    # Convert history to be JSON serializable
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(config.TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f)

    print("Training complete. Final model and history saved.")
    return model, history

def save_model_metadata(model, X_train, y_train, history):
    print("\n--- Saving model metadata ---")
    metadata = {
        "model_name": "health_lstm_model",
        "version": "2.0", # Updated version for the new dataset
        "training_timestamp": str(datetime.now()),
        "training_data_shape": {
            "X_train": X_train.shape,
            "y_train": y_train.shape
        },
        "model_params": model.count_params(),
        "training_metrics": {
            "final_loss": history.history["loss"][-1],
            "final_accuracy": history.history["accuracy"][-1],
            "final_val_loss": history.history["val_loss"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1]
        }
    }
    with open(config.MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to '{config.MODEL_METADATA_PATH}'")

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)

    # Run the pipeline with the new procedural data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_prepare_data()
    model = build_model()
    trained_model, history = run_training(model, X_train, y_train, X_val, y_val)
    
    # Save metadata
    save_model_metadata(trained_model, X_train, y_train, history)
    
    print("\nNew model training finished successfully!")