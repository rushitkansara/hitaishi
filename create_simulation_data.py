import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_gen')))
from data_gen.datagen import classes

X_test = np.load('data_gen/data/procedural_X_test.npz')['X_test']
y_test = np.load('data_gen/data/procedural_y_test.npy')

emergency_classes = range(2, 13) # Range of emergency classes
simulation_data = []

for class_label in emergency_classes:
    sample_indices = np.where(y_test == class_label)[0]
    if len(sample_indices) > 0:
        sample_index = sample_indices[0]
        sequence = X_test[sample_index].tolist()
        simulation_data.append({
            "patient_name": f"Simulated Patient - {classes[class_label]}",
            "class_label": class_label,
            "sequence": sequence
        })

with open('web/src/data/simulationData.js', 'w') as f:
    f.write(f"export const simulationData = {json.dumps(simulation_data, indent=2)};")

print("Successfully created simulationData.js")
