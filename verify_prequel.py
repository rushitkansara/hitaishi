import numpy as np
from src.ml.simulation_manager import SimulationManager
import config

sm = SimulationManager(config.MODEL_V1_2_H5_PATH)
sm.initialize_patient("prequel_test", "PrequelTester", 24, "male", ["Healthy"])

# Fetch the first tick immediately after init
tick = sm.get_next_tick("prequel_test")

if "history" in tick:
    print(f"SUCCESS: Prequel data received. History length: {len(tick['history'])}")
    print(f"First 3 points of HR: {[h['heart_rate'] for h in tick['history'][:3]]}")
else:
    print("FAILURE: No prequel history in first tick!")
