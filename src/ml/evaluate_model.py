
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def evaluate_model(model, X_test, y_test, class_names):
    print("\n--- Task 2.4: Evaluating Model ---")
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred_indices = y_pred_probs.argmax(axis=1)
    y_test_indices = y_test # y_test is now integer encoded

    # Overall metrics
    print("="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    accuracy = (y_pred_indices == y_test_indices).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE")
    print("="*60)
    report = classification_report(y_test_indices, y_pred_indices, target_names=class_names)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test_indices, y_pred_indices)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png', dpi=300)
    print("\nConfusion matrix saved to 'models/confusion_matrix.png'")

    # Critical class performance
    critical_classes_indices = [i for i, name in enumerate(class_names) if name not in ['Stable', 'Monitor']]
    critical_mask = np.isin(y_test_indices, critical_classes_indices)
    
    critical_accuracy = 0
    if critical_mask.sum() > 0:
        critical_accuracy = (y_pred_indices[critical_mask] == y_test_indices[critical_mask]).mean()
        print(f"\nCritical Emergency Detection Accuracy: {critical_accuracy:.4f}")

        print("\nRecall for Critical Classes:")
        for cls_idx in critical_classes_indices:
            cls_mask = y_test_indices == cls_idx
            if cls_mask.sum() > 0:
                recall = (y_pred_indices[cls_mask] == cls_idx).mean()
                print(f"  - {class_names[cls_idx]} Recall: {recall:.4f}")

    # Save evaluation report
    report_path = 'models/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("Hitaishi Model Evaluation Report\n")
        f.write("="*30 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Critical Emergency Detection Accuracy: {critical_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Evaluation report saved to '{report_path}'")

    return y_pred_indices, y_pred_probs

def main():
    import pickle
    # Class names from the spec
    class_names = [
        'Stable', 'Monitor', 'Heart_Attack', 'Arrhythmia',
        'Heart_Failure', 'Hypoglycemia', 'Hyperglycemia_DKA',
        'Respiratory_Distress', 'Sepsis', 'Stroke',
        'Shock', 'Hypertensive_Crisis', 'Fall_Unconscious'
    ]

    # Load the trained model
    print("Loading trained model: 'models/best_model.h5'")
    # Load the best model for a more reliable evaluation
    model = tf.keras.models.load_model('models/best_model.h5')

    # Load the test data
    print("Loading test data...")
    X_test = np.load('data-gen/data/X_test.npy')
    y_test = np.load('data-gen/data/y_test.npy')

    # Load the scaler and normalize the test data
    scaler_path = 'data-gen/data/scaler_params.pkl'
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X_test = scaler.transform(X_test.reshape(-1, 8)).reshape(X_test.shape)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, class_names)
    
    print("\nEvaluation script finished.")

if __name__ == '__main__':
    main()
