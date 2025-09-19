
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
from typing import List, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from src.utils import ROOT

def _load_pred(file: str):
    df = pd.read_csv(file)
    if df.shape[1] == 1:
        label_col = df.columns[0]; ids = None
    else:
        ids = df.columns[0]; label_col = df.columns[-1]
    return df, ids, label_col

def _load_oof(file: str):
    df = pd.read_csv(file)
    y = df["y"].values if "y" in df.columns else None
    if "oof_prob" in df.columns: x = df["oof_prob"].values
    elif "oof_pred" in df.columns: x = df["oof_pred"].values
    else: x = df.iloc[:, -1].values
    return x, y

def run_stack_learn(test_pred_files: List[str], oof_files: Optional[List[str]] = None,
                    task: str = "regression", meta: str = "ridge", out_name: Optional[str] = None):
    assert len(test_pred_files) >= 2, "need >=2 files"
    dfs, labels, id_col = [], [], None
    for f in test_pred_files:
        df, ids, lbl = _load_pred(f)
        if id_col is None and ids is not None: id_col = ids
        dfs.append(df); labels.append(lbl)
    X_test = np.column_stack([df[lbl].values for df,lbl in zip(dfs, labels)])
    if task == "regression":
        meta_model = LGBMRegressor(n_estimators=500, learning_rate=0.05) if meta == "lgbm" else Ridge(alpha=1.0)
    else:
        meta_model = LGBMClassifier(n_estimators=500, learning_rate=0.05) if meta == "lgbm" else LogisticRegression(max_iter=1000)
    if oof_files and len(oof_files) == len(test_pred_files):
        X_oof_parts, y_ref = [], None
        for f in oof_files:
            x, y = _load_oof(f); X_oof_parts.append(x.reshape(-1,1))
            if y_ref is None and y is not None: y_ref = y
        X_oof = np.hstack(X_oof_parts)
        if y_ref is not None: meta_model.fit(X_oof, y_ref); y_pred = meta_model.predict_proba(X_test)[:,1] if (task=="classification" and hasattr(meta_model,"predict_proba")) else meta_model.predict(X_test)
        else: y_pred = X_test.mean(axis=1)
    else:
        y_pred = X_test.mean(axis=1)
    base = dfs[0]
    out = base[[id_col]].copy() if (id_col and id_col in base.columns) else pd.DataFrame()
    label = labels[0]; out[label] = y_pred
    out_dir = ROOT / "outputs" / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (out_name or f"stack_learn_{len(test_pred_files)}.csv")
    out.to_csv(path, index=False); return path
