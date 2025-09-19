
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from src.utils import ROOT

def _detect_cols(df: pd.DataFrame):
    if df.shape[1] == 1: return None, df.columns[0]
    return df.columns[0], df.columns[-1]

def _load_oof(path: Path):
    df = pd.read_csv(path)
    y = df["y"].values if "y" in df.columns else None
    if "oof_prob" in df.columns: x = df["oof_prob"].values
    elif "oof_pred" in df.columns: x = df["oof_pred"].values
    else: x = df.iloc[:, -1].values
    return x, y

def run_stack_cv(run_ids: List[str], cfg: dict, task: str = "regression",
                 meta: str = "ridge", label: Optional[str] = None, out_name: Optional[str] = None):
    out_dir = ROOT / cfg["output"]["dir"]; oof_dir = out_dir / "oof"; preds_dir = out_dir / "preds"
    X_oof_parts = []; y_ref = None
    for rid in run_ids:
        x, y = _load_oof(oof_dir / f"{rid}.csv"); X_oof_parts.append(x.reshape(-1,1)); y_ref = y_ref if y_ref is not None else y
    X_oof = np.hstack(X_oof_parts)
    if task == "regression":
        meta_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05) if meta == "lgbm" else Ridge(alpha=1.0)
    else:
        meta_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05) if meta == "lgbm" else LogisticRegression(max_iter=1000)
    if y_ref is not None: meta_model.fit(X_oof, y_ref)
    test_frames = [pd.read_csv(preds_dir / f"{rid}.csv") for rid in run_ids]
    id_col, label_col = _detect_cols(test_frames[0]); label = label or label_col
    X_test = np.column_stack([df[label_col].values for df in test_frames])
    if y_ref is None: y_pred = X_test.mean(axis=1)
    else: y_pred = meta_model.predict_proba(X_test)[:,1] if (task=="classification" and hasattr(meta_model, "predict_proba")) else meta_model.predict(X_test)
    base = test_frames[0]; sub = base[[id_col]].copy() if id_col and id_col in base.columns else pd.DataFrame()
    sub[label] = y_pred; sub_path = preds_dir / (out_name or f"stack_cv_{len(run_ids)}_{meta}.csv")
    sub.to_csv(sub_path, index=False)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "stack_cv_last.json").write_text(json.dumps({"task": task, "meta": meta, "run_ids": run_ids, "used_label_col": label_col, "output": str(sub_path)}, indent=2), encoding="utf-8")
    return sub_path
