# src/models.py
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))

def build_model(name: str, params: dict = None):
    params = params or {}
    if name == "hgb":
        base = dict(learning_rate=0.05, random_state=42)
        base.update(params)
        return HistGradientBoostingRegressor(**base)
    if name == "xgb":
        base = dict(learning_rate=0.05, random_state=42)
        base.update(params)
        return XGBRegressor(**base)
    if name == "lgbm":
        base = dict(learning_rate=0.05, random_state=42)
        base.update(params)
        return LGBMRegressor(**base)
    raise ValueError("Modelo no soportado")
