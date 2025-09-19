
from __future__ import annotations

class ModelRegistry:
    def __init__(self, task: str):
        self.task = task

    def build(self, name: str, params: dict):
        name = name.lower()
        if name == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
            return LGBMClassifier(**params) if self.task == "classification" else LGBMRegressor(**params)
        if name == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            base = dict(tree_method="hist", n_estimators=600); base.update(params or {})
            return XGBClassifier(**base) if self.task == "classification" else XGBRegressor(**base)
        if name == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            base = dict(verbose=False); base.update(params or {})
            return CatBoostClassifier(**base) if self.task == "classification" else CatBoostRegressor(**base)
        if name == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(**(params or {}))
        if name == "rf":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(**(params or {})) if self.task == "classification" else RandomForestRegressor(**(params or {}))
        raise ValueError(f"Unknown model: {name}")
