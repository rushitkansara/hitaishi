import numpy as np
import json

X_test = np.load('data-gen/data/X_test.npy')
y_test = np.load('data-gen/data/y_test.npy')

emergency_classes = range(2, 14)
simulation_data = []

class_names = [
    'Stable', 'Monitor', 'Heart_Attack', 'Arrhythmia',
    'Heart_Failure', 'Hypoglycemia', 'Hyperglycemia_DKA',
    'Respiratory_Distress', 'Sepsis', 'Stroke',
    'Shock', 'Hypertensive_Crisis', 'Fall_Unconscious'
]

for class_label in emergency_classes:
    sample_indices = np.where(y_test == class_label)[0]
    if len(sample_indices) > 0:
        sample_index = sample_indices[0]
        sequence = X_test[sample_index].tolist()
        simulation_data.append({
            "patient_name": f"Simulated Patient - {class_names[class_label]}",
            "class_label": class_label,
            "sequence": sequence
        })

with open('web/src/data/simulationData.js', 'w') as f:
    f.write(f"export const simulationData = {json.dumps(simulation_data, indent=2)};")

print("Successfully created simulationData.js")
