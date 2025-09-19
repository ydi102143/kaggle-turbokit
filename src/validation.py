
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

def make_splits(task: str, y, groups=None, method="kfold", folds=5, shuffle=True, random_state=42):
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
    splitter = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(np.zeros(len(y))))
