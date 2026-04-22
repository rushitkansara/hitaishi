
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os
from datetime import datetime
import json

def load_v2_data(data_path: str = 'data_v2/'):
    """Loads and prepares the 600s window dataset."""
    print(f"Loading dataset from {data_path}...")
    train_data = np.load(os.path.join(data_path, 'v2_train.npz'))
    test_data = np.load(os.path.join(data_path, 'v2_test.npz'))
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    def fill_nans(arr):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[2]):
                series = arr[i, :, j]
                mask = np.isnan(series)
                if np.any(mask):
                    idx = np.where(~mask)[0]
                    if len(idx) > 0:
                        series[mask] = np.interp(np.where(mask)[0], idx, series[idx])
                    else:
                        series[mask] = 0.0
        return arr
    
    print("Repairing telemetry artifacts (NaNs)...")
    X_train = fill_nans(X_train)
    X_test = fill_nans(X_test)
    
    return X_train, y_train, X_test, y_test

def build_v2_lstm(input_shape=(600, 7), num_classes=13):
    """Deep LSTM architecture for 10-minute clinical windows."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    # 1. Setup Isolated Experiment Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"v2_lstm_{timestamp}"
    exp_dir = os.path.join('..', 'experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"--- INITIALIZING EXPERIMENT: {exp_name} ---")
    print(f"Artifacts will be saved to: {exp_dir}")

    # 2. Load Data
    X_train, y_train, X_test, y_test = load_v2_data()
    
    # 3. Build Model
    model = build_v2_lstm()
    
    # 4. Training Callbacks (Experiment Aware)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(os.path.join(exp_dir, 'best_model.keras'), save_best_only=True),
        TensorBoard(log_dir=os.path.join(exp_dir, 'logs'))
    ]
    
    # 5. Train
    print(f"\n--- Training V2.0 Model on {len(y_train)} samples ---")
    print("Ceiling set to 500 epochs. EarlyStopping will terminate when convergence is reached.")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2, # Matching 70/30 split logic
        epochs=500,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluation & Final Artifacts
    print("\n--- Running Final Evaluation ---")
    loss, acc = model.evaluate(X_test, y_test)
    
    # Save History and Metrics
    results = {
        'test_accuracy': float(acc),
        'test_loss': float(loss),
        'training_history': history.history
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save final model weights
    model.save(os.path.join(exp_dir, 'final_model.h5'))
    
    print(f"\nTraining Complete!")
    print(f"Best model and logs saved in: {exp_dir}")
    print(f"Final Test Accuracy: {acc:.4f}")
