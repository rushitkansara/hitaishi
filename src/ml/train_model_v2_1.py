
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
from datetime import datetime
import json
from evaluation_utils import generate_evaluation_report

CLASS_NAMES = [
    'Stable', 'Monitor', 'Heart_Attack', 'Arrhythmia', 'Heart_Failure',
    'Hypoglycemia', 'Hyperglycemia_DKA', 'Respiratory_Distress', 'Sepsis',
    'Stroke', 'Shock', 'Hypertensive_Crisis', 'Fall_Unconscious'
]

def load_v2_data(data_path: str = 'data_v2/'):
    """Loads the 600s window dataset."""
    print(f"Loading dataset from {data_path}...")
    train_data = np.load(os.path.join(data_path, 'v2_train.npz'))
    test_data = np.load(os.path.join(data_path, 'v2_test.npz'))
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    def fill_nans(arr):
        # Repair telemetry loss via interpolation
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

def build_v2_1_hybrid(input_shape=(600, 7), num_classes=13):
    """
    Physiological Hybrid v2.1: 
    Full-resolution Parallel CNN + Bi-LSTM + Attention.
    """
    inputs = Input(shape=input_shape)

    # --- PATH A: Micro-Scale (Second-to-second jitter) ---
    path_a = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    path_a = layers.BatchNormalization()(path_a)

    # --- PATH B: Macro-Scale (Minute-to-minute trends) ---
    path_b = layers.Conv1D(64, kernel_size=5, padding='same', dilation_rate=4, activation='relu')(inputs)
    path_b = layers.BatchNormalization()(path_b)

    # --- Merge Paths ---
    merged = layers.Concatenate()([path_a, path_b])
    merged = layers.SpatialDropout1D(0.2)(merged)

    # --- Global Temporal Context (Bidirectional) ---
    lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(merged)
    lstm_out = layers.LayerNormalization()(lstm_out)

    # --- Attention Mechanism (Relevance Map) ---
    attention_out = layers.MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)(lstm_out, lstm_out)
    attention_out = layers.Add()([lstm_out, attention_out]) # Residual connection
    attention_out = layers.LayerNormalization()(attention_out)

    # --- Classification Head ---
    # Global Max Pooling picks up the "strongest" clinical signal in the window
    avg_pool = layers.GlobalAveragePooling1D()(attention_out)
    max_pool = layers.GlobalMaxPooling1D()(attention_out)
    head = layers.Concatenate()([avg_pool, max_pool])
    
    head = layers.Dense(128, activation='relu')(head)
    head = layers.Dropout(0.4)(head)
    head = layers.Dense(64, activation='relu')(head)
    outputs = layers.Dense(num_classes, activation='softmax')(head)

    model = Model(inputs=inputs, outputs=outputs)
    
    # Cosine Decay Learning Rate Scheduler
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=50000 # Approximately 50-70 epochs
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    # 1. Setup Isolated Experiment Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use hitaishi-alpha for the new architecture experiment
    exp_name = f"v2_1_hybrid_tuned_{timestamp}" # Tuned version name
    # Update exp_dir to be under hitaishi-alpha
    exp_dir = os.path.join('..', 'experiments', 'hitaishi-alpha', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"--- INITIALIZING TUNING EXPERIMENT: {exp_name} ---")
    print(f"Artifacts will be saved to: {exp_dir}")

    # 2. Load Data using tf.data optimized loader
    # We will get numpy arrays back and let Keras Tuner handle batching and splitting.
    X_train_np, y_train_np, X_test_np, y_test_np = load_v2_data_tf(shuffle=True, buffer_size=10000)

    # 3. Initialize Hyperband Tuner (efficient for shorter searches)
    # We will tune batch size, learning rate, and model architecture hyperparameters.
    # Let's set a reasonable max_epochs for the tuner search to keep it manageable.
    max_epochs_tuner = 100 # Max epochs for each trial during search
    # Using RandomSearch for broader exploration first, can switch to Hyperband if speed is critical.
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=20, # Number of different hyperparameter combinations to try
        executions_per_trial=1, # Number of models to train for each trial
        directory=exp_dir, # Save tuner search logs within the hitaishi-alpha experiment dir
        project_name='keras_tuner_search',
        # Disable saving tuner logs to keep experiment directory clean if desired, or keep for debugging
        # overwrite=True # Use if re-running
    )

    # 4. Perform Hyperparameter Search
    print("\n--- Starting Hyperparameter Search ---")
    # Keras Tuner will handle validation split internally if validation_data is not provided
    # Or we can pre-split if needed. For simplicity, let's rely on tuner's default behavior or provide X_test, y_test as validation set.
    # Let's use a portion of training data for validation during tuning for efficiency.
    
    # Define a simple validation split for the tuner search
    val_split = 0.2
    num_train_samples = len(X_train_np)
    num_val_samples = int(num_train_samples * val_split)
    
    X_train_tuner = X_train_np[:-num_val_samples]
    y_train_tuner = y_train_np[:-num_val_samples]
    X_val_tuner = X_train_np[-num_val_samples:]
    y_val_tuner = y_train_np[-num_val_samples:]

    print(f"Tuning with {len(y_train_tuner)} training samples and {len(y_val_tuner)} validation samples.")

    tuner.search(
        X_train_tuner, y_train_tuner,
        epochs=max_epochs_tuner,
        validation_data=(X_val_tuner, y_val_tuner),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        # Note: ModelCheckpoint is usually used for saving the *best* model during a single run,
        # Keras Tuner handles saving the best trial's model separately.
    )

    # 5. Get Best Hyperparameters and Model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best hyperparameters found: {best_hps.values}")

    # Build the final model with the best hyperparameters
    print("\n--- Building Final Model with Best Hyperparameters ---")
    final_model = tuner.hypermodel.build(best_hps)
    final_model.summary()

    # We need to re-compile the model if it was built but not compiled by tuner or if we want to retrain.
    # Keras Tuner's `get_best_models` returns models that might not be re-trainable directly without recompiling.
    # It's better to re-train from scratch with the best HPs.
    
    # Re-create the dataset for full training with the best batch size found
    best_batch_size = best_hps.get('batch_size') or 32 # Use default if not tuned
    print(f"\nRetraining final model with batch_size: {best_batch_size}")
    
    # Create final training and validation datasets with the best batch size
    final_train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
    final_train_dataset = final_train_dataset.shuffle(buffer_size=10000).batch(best_batch_size).prefetch(tf.data.AUTOTUNE)
    
    final_val_dataset = tf.data.Dataset.from_tensor_slices((X_val_tuner, y_val_tuner)) # Use the same validation split
    final_val_dataset = final_val_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

    # Re-compile the model (important for fresh training)
    learning_rate = best_hps.get('learning_rate') or 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    final_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks for the final training run
    # Save final training in a subdirectory within the main experiment dir
    final_exp_dir = os.path.join(exp_dir, 'final_training') 
    os.makedirs(final_exp_dir, exist_ok=True)
    
    final_callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), # Increased patience for full run
        ModelCheckpoint(os.path.join(final_exp_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss'),
        TensorBoard(log_dir=os.path.join(final_exp_dir, 'logs')),
    ]
    
    # Train the final model
    print("\n--- Training Final Model with Best Hyperparameters ---")
    # The number of epochs for the final run can be set higher as we use early stopping
    final_history = final_model.fit(
        final_train_dataset,
        validation_data=final_val_dataset,
        epochs=500, # Set a large number, rely on early stopping
        callbacks=final_callbacks,
        verbose=1
    )
    
    # 6. Final Evaluation and Reporting
    print("\n--- Running Final Evaluation on Test Set ---")
    # Load the best model found by tuner for evaluation, or use the final_model after full training.
    # Using the model trained with best HPs on full data + early stopping.
    
    # Evaluate the final trained model on the separate test set
    test_loss, test_acc = final_model.evaluate(X_test_np, y_test_np) # Use numpy arrays directly for evaluate

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Generate Detailed Report
    report_dict = generate_evaluation_report(
        model=final_model,
        X_test=X_test_np, # Pass numpy array for evaluation
        y_test=y_test_np, # Pass numpy array for evaluation
        history=final_history, # Use history from the final training run
        exp_dir=final_exp_dir, # Save report in the final training directory
        class_names=CLASS_NAMES
    )
    
    # Save final results including best HPs and classification report
    results = {
        'best_hyperparameters': best_hps.values,
        'final_training_test_accuracy': float(test_acc),
        'final_training_test_loss': float(test_loss),
        'final_training_history': final_history.history,
        'classification_report': report_dict
    }
    with open(os.path.join(final_exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    final_model.save(os.path.join(final_exp_dir, 'final_tuned_model.h5'))
    print(f"\nTraining and Tuning Complete!")
    print(f"Best hyperparameters saved in: {final_exp_dir}/results.json")
    print(f"Detailed report saved in: {final_exp_dir}/REPORT.md")
