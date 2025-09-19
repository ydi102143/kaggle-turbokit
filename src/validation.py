from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn>=1.3
except Exception:
    StratifiedGroupKFold = None

def make_splits(task: str, y, groups=None, method="kfold", folds=5, shuffle=True, random_state=42, regression_stratify_bins: int = 10):
    if method in ["stratified", "stratifiedkfold"] or (task == "classification" and method == "kfold"):
        splitter = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)), y))
    if method == "group":
        assert groups is not None, "groups required for GroupKFold"
        splitter = GroupKFold(n_splits=folds)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    if method == "time":
        splitter = TimeSeriesSplit(n_splits=folds)
        return list(splitter.split(np.arange(len(y))))
    if method in ["group_stratified", "stratified_group", "sgkf"]:
        if StratifiedGroupKFold is not None and task == "classification":
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            return list(splitter.split(np.zeros(len(y)), y, groups))
        splitter = GroupKFold(n_splits=folds)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    if task != "classification" and method in ["stratified", "stratifiedkfold"]:
        y = np.array(y)
        q = max(2, regression_stratify_bins)
        bins = np.unique(np.quantile(y, np.linspace(0, 1, q+1)))
        yb = np.digitize(y, bins[1:-1], right=True)
        splitter = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)), yb))
    splitter = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(np.zeros(len(y))))
