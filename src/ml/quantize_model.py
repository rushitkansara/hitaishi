import tensorflow as tf
import numpy as np
import os
import time

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Import variables from the data generation script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data_gen')))
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
            yield [sample.reshape(1, 60, 7).astype(np.float32)]

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
        sample = X_test[i % len(X_test)].reshape(1, 60, 7).astype(np.float32)

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
    
    # Load procedural training data for representative dataset
    X_train_full = np.load(config.PROCEDURAL_X_TRAIN_PATH)['X_train']
    y_train_full = np.load(config.PROCEDURAL_Y_TRAIN_PATH)

    # Load procedural test data for benchmarking
    X_test_full = np.load(config.PROCEDURAL_X_TEST_PATH)['X_test']
    y_test_full = np.load(config.PROCEDURAL_Y_TEST_PATH)

    # Load the procedural scaler
    with open(config.PROCEDURAL_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Normalize X_train for representative dataset (only a subset is needed)
    X_train_normalized = scaler.transform(X_train_full.reshape(-1, 7)).reshape(X_train_full.shape)
    
    # Normalize X_test for benchmarking
    X_test_normalized = scaler.transform(X_test_full.reshape(-1, 7)).reshape(X_test_full.shape)

    # Use a subset of X_train_normalized for the representative dataset
    # This is important for full integer quantization
    X_train_representative = X_train_normalized[np.random.choice(X_train_normalized.shape[0], 1000, replace=False)]

    model_path = config.BEST_MODEL_PATH # Quantize the best performing model
    quantized_model_path = quantize_model(model_path, X_train_representative)
    
    # Benchmark the quantized model
    benchmark_inference(quantized_model_path, X_test_normalized)
    
    print("\nQuantization script finished.")

if __name__ == '__main__':
    import pickle
    from sklearn.model_selection import train_test_split
    main()