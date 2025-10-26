import tensorflow as tf
import numpy as np
import os
import time

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Import variables from the data generation script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data-gen')))
from datagen import classes


def quantize_model(model_path, X_train):
    print("\n--- Task 2.5: Quantizing Model (INT8) ---")
    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Save the model in SavedModel format
    saved_model_dir = "/tmp/saved_model"
    model.export(saved_model_dir)

    # Create representative dataset for quantization
    def representative_dataset():
        for i in range(100):
            sample = X_train[np.random.randint(0, len(X_train))]
            yield [sample.reshape(1, 60, 8).astype(np.float32)]

    # Convert to TFLite with INT8 quantization from SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    quantized_tflite_model = converter.convert()

    # Save quantized model
    quantized_model_path = config.HEALTH_MODEL_QUANTIZED_PATH
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_tflite_model)

    # Compare sizes
    original_size = os.path.getsize(model_path) / 1024
    quantized_size = len(quantized_tflite_model) / 1024

    print(f"Original model size: {original_size:.2f} KB")
    print(f"Quantized model size: {quantized_size:.2f} KB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    return quantized_model_path

def benchmark_inference(model_path, X_test, num_runs=100):
    print("\n--- Benchmarking Inference Speed ---")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    times = []
    for i in range(num_runs):
        sample = X_test[i % len(X_test)].reshape(1, 60, 8).astype(np.float32)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    print(f"Inference Benchmark ({num_runs} runs):")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")

def main():
    # We need X_train for the representative dataset and X_test for benchmarking
    print("Loading data for quantization and benchmarking...")
    
    X_base = np.load(config.TRAINING_SEQUENCES_PATH)['X_train']
    X_aug = np.load(config.AUGMENTED_SEQUENCES_PATH)['X_aug']
    X_full = np.concatenate([X_base, X_aug], axis=0)
    
    y_base = np.load(config.TRAINING_LABELS_PATH)
    y_aug = np.load(config.AUGMENTED_LABELS_PATH)
    y_full = np.concatenate([y_base, y_aug], axis=0)

    with open(config.SCALER_PARAMS_PATH, 'rb') as f:
        scaler = pickle.load(f)
    original_shape = X_full.shape
    X_reshaped = X_full.reshape(-1, original_shape[2])
    X_normalized = scaler.transform(X_reshaped)
    X_full_normalized = X_normalized.reshape(original_shape)

    X_train, X_temp, _, _ = train_test_split(
        X_full_normalized, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    
    X_test = np.load(config.X_TEST_PATH)
    # Normalize X_test for benchmarking
    X_test = scaler.transform(X_test.reshape(-1, 8)).reshape(X_test.shape)

    model_path = config.HEALTH_LSTM_MODEL_PATH
    quantized_model_path = quantize_model(model_path, X_train)
    
    # Benchmark the quantized model
    benchmark_inference(quantized_model_path, X_test)
    
    print("\nQuantization script finished.")

if __name__ == '__main__':
    import pickle
    from sklearn.model_selection import train_test_split
    main()