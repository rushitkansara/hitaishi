import numpy as np
from src.ml.simulation_manager import SimulationManager
import config

sm = SimulationManager(config.MODEL_V1_2_H5_PATH)
sm.initialize_patient("debug_30", "Debug", 30, "male", ["Healthy"])
# Check if tick 0 has history if requested
tick = sm.get_next_tick("debug_30", request_history=True)
print(f"Has history: {'history' in tick}")
print(f"Total ticks: {sm.active_sessions['debug_30']['total_ticks']}")
