
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random

# Import variables from the data generation script
from datagen import classes, features

def visualize_samples(input_dir='data/', num_samples=3, output_file='sample_visualizations.png'):
    """Visualize some samples from the generated dataset."""
    print("--- Generating Sample Visualizations ---")

    # Load data
    try:
        X_train = np.load(os.path.join(input_dir, 'training_sequences.npz'))['X_train']
        y_train = np.load(os.path.join(input_dir, 'training_labels.npy'))
    except FileNotFoundError:
        print(f"Error: Could not find training data in '{input_dir}'. Please run the data generator first.")
        return

    # Create a figure with subplots
    # One row for each class, one column for each sample
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(num_samples * 6, len(classes) * 4), squeeze=False)
    fig.suptitle('Sample Visualizations of Generated Time-Series Data', fontsize=16)

    for class_label, class_name in classes.items():
        # Find indices for the current class
        class_indices = np.where(y_train == class_label)[0]
        if len(class_indices) == 0:
            print(f"Warning: No samples found for class '{class_name}'. Skipping.")
            for j in range(num_samples):
                axes[class_label, j].text(0.5, 0.5, 'No Samples', ha='center', va='center')
                axes[class_label, j].set_title(f"Class: {class_name}")
            continue

        # Randomly select samples to plot
        sample_indices = random.sample(list(class_indices), min(num_samples, len(class_indices)))

        for j, sample_index in enumerate(sample_indices):
            ax = axes[class_label, j]
            # Plot each feature with a label
            for i in range(X_train.shape[2]):
                ax.plot(X_train[sample_index, :, i], label=features[i])
            
            ax.set_title(f"Class: {class_name} (Sample #{j+1})")
            if class_label == len(classes) - 1:
                ax.set_xlabel("Time (seconds)")
            if j == 0:
                ax.set_ylabel("Scaled Value")

    # Add a single legend to the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    save_path = os.path.join(input_dir, output_file)
    plt.savefig(save_path)
    print(f"--- Visualizations saved to {save_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize samples from the synthetic medical time-series dataset.')
    parser.add_argument('--input', type=str, default='data/', help='Input directory for generated files.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize per class.')
    parser.add_argument('--output', type=str, default='sample_visualizations.png', help='Output file for the visualizations.')
    args = parser.parse_args()

    visualize_samples(args.input, args.num_samples, args.output)
