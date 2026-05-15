import numpy as np
from src.ml.simulation_manager import SimulationManager
import config

# Initialize Manager
sm = SimulationManager(config.MODEL_V1_2_H5_PATH)
sm.initialize_patient("test_robust", "RobustTester", 30, "male", ["Healthy"])

print("--- PHASE 1: CHECKING FIRST 10 TICKS (PADDING VERIFICATION) ---")
for i in range(10):
    tick = sm.get_next_tick("test_robust")
    print(f"Tick {i}: Predicted={tick['prediction']['label']}, Rationale={tick['prediction']['rationale']}")

print("\n--- PHASE 2: TRIGGERING EMERGENCY 8 (SEPSIS) ---")
sm.trigger_emergency("test_robust", 8)

for i in range(40):
    tick = sm.get_next_tick("test_robust")
    if i % 10 == 0 or i == 39:
        print(f"Tick {i}: Predicted={tick['prediction']['label']}, Confidence={tick['prediction']['confidence']:.2f}")

