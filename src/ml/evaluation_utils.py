
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import io
import pandas as pd

def generate_evaluation_report(model, X_test, y_test, history, exp_dir, class_names):
    """
    Generates a comprehensive evaluation report and saves artifacts to exp_dir.
    """
    print("\n--- Generating Detailed Evaluation Report ---")
    
    # 1. Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 2. Overall Metrics
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Hitaishi V2.1 Hybrid')
    plt.ylabel('True Clinical State')
    plt.xlabel('Predicted Clinical State')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 4. Training History Plots
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    plt.close()
    
    # 5. Model Architecture Summary (captured as string)
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    model_summary_str = stream.getvalue()
    
    # 6. Generate Markdown Report
    report_md_path = os.path.join(exp_dir, 'REPORT.md')
    with open(report_md_path, 'w') as f:
        f.write("# Hitaishi V2.1 Model Evaluation Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Overall Accuracy:** {report_dict['accuracy']:.4f}\n\n")
        
        f.write("## 1. Executive Summary\n")
        critical_recall = np.mean([report_dict[cls]['recall'] for cls in class_names if cls not in ['Stable', 'Monitor']])
        f.write(f"- **Critical Emergency Recall (Mean):** {critical_recall:.4f}\n")
        f.write("- **Analysis:** High recall in critical classes is prioritized to minimize false negatives in life-threatening scenarios.\n\n")
        
        f.write("## 2. Training Performance\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        
        f.write("## 3. Classification Detailed Report\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n\n")
        
        f.write("## 4. Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("## 5. Model Architecture & Layer Details\n")
        f.write("```\n")
        f.write(model_summary_str)
        f.write("\n```\n\n")
        
        f.write("## 6. Layer-by-Layer Parameters\n")
        f.write("| Layer Name | Type | Output Shape | Param # |\n")
        f.write("|------------|------|--------------|---------|\n")
        for layer in model.layers:
            # Skip InputLayer as it does not have output_shape or count_params in the same way computational layers do.
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            
            try:
                output_shape_str = str(layer.output_shape)
                param_count = layer.count_params()
                f.write(f"| {layer.name} | {type(layer).__name__} | {output_shape_str} | {param_count} |\n")
            except AttributeError:
                # Fallback for any other layers that might not have these attributes
                f.write(f"| {layer.name} | {type(layer).__name__} | N/A | N/A |\n")
            
    print(f"Report generated successfully at: {report_md_path}")
    return report_dict
