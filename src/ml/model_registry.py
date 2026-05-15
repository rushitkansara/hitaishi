import os
import config
from src.ml.inference_engine import HealthRiskPredictor

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        MODEL_PATH = config.MODEL_V1_2_H5_PATH
        SCALER_PATH = config.DATA_PROCEDURAL_SCALER_PATH
        _predictor = HealthRiskPredictor(MODEL_PATH, SCALER_PATH)
    return _predictor
