from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def _prep(train: pd.DataFrame, test: pd.DataFrame, target: str | None):
    common = [c for c in train.columns if c in test.columns]
    if target and target in common:
        common.remove(target)
    X = pd.concat([train[common].copy(), test[common].copy()], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    return X, y, common

def run_adversarial_validation(cfg: dict) -> dict:
    train = pd.read_csv(Path(cfg["data"]["train"]).resolve())
    test = pd.read_csv(Path(cfg["data"]["test"]).resolve())
    target = cfg["data"].get("target")
    X, y, _ = _prep(train, test, target)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse=True))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("est", model)])
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xva)[:,1]
    auc = roc_auc_score(yva, prob)
    return {"auc": float(auc), "top_features": []}
