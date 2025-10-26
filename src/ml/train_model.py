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
    print("--- Task 2.1: Loading and Preparing Data ---")
    
    # Load base training data
    X_base = np.load(config.TRAINING_SEQUENCES_PATH)['X_train']
    y_base = np.load(config.TRAINING_LABELS_PATH)
    
    # Load augmented data
    X_aug = np.load(config.AUGMENTED_SEQUENCES_PATH)['X_aug']
    y_aug = np.load(config.AUGMENTED_LABELS_PATH)
    
    # Combine
    X_full = np.concatenate([X_base, X_aug], axis=0)
    y_full = np.concatenate([y_base, y_aug], axis=0)
    print(f"Combined data shape: X={X_full.shape}, y={y_full.shape}")
    
    # Split first to keep a raw version of X_test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=0.1, stratify=y_full, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, stratify=y_train_val, random_state=42 # 0.111 * 0.9 = 0.1
    )

    # Load and fit scaler on training data only
    with open(config.SCALER_PARAMS_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Normalize the training and validation sets
    X_train = scaler.transform(X_train.reshape(-1, 8)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, 8)).reshape(X_val.shape)

    print(f"  X_train.shape = {X_train.shape}")
    print(f"  y_train.shape = {y_train.shape}")
    print(f"  X_val.shape = {X_val.shape}")
    print(f"  y_val.shape = {y_val.shape}")
    print(f"  X_test.shape = {X_test.shape} (raw)")
    print(f"  y_test.shape = {y_test.shape}")
    print("Data preparation complete.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Task 2.2: Define LSTM Architecture
def build_model():
    print("\n--- Task 2.2: Building LSTM Model ---")
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(60, 8)),
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

def save_test_data(X_test, y_test):
    print("\n--- Saving test data for later evaluation ---")
    # We need to save X_test and y_test for the evaluation script.
    # Let's save them in the data-gen/data directory as well.
    np.save(config.X_TEST_PATH, X_test)
    np.save(config.Y_TEST_PATH, y_test)
    print(f"Test data saved to '{config.X_TEST_PATH}' and '{config.Y_TEST_PATH}'")

def save_model_metadata(model, X_train, y_train, history):
    print("\n--- Saving model metadata ---")
    metadata = {
        "model_name": "health_lstm_model",
        "version": "1.0",
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

    # Run the pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_prepare_data()
    model = build_model()
    trained_model, history = run_training(model, X_train, y_train, X_val, y_val)
    
    # The evaluation script will need the test set. Let's save it.
    save_test_data(X_test, y_test)
    save_model_metadata(trained_model, X_train, y_train, history)
    
    print("\nSprint 2 - Training script finished successfully!")