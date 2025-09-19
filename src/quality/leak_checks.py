
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def _safe_auc(y, x):
    try:
        if len(np.unique(y)) < 2 or len(np.unique(x)) < 2: return None
        return float(roc_auc_score(y, x))
    except Exception: return None

def _corr_numeric(y: pd.Series, x: pd.Series) -> float | None:
    try:
        if x.dtype.kind in "biufc" and x.nunique(dropna=True) > 1:
            return float(y.corr(x))
        return None
    except Exception: return None

def _leak_by_dup_keys(train: pd.DataFrame, test: pd.DataFrame, keys: List[str]) -> Dict[str, Any]:
    if not keys: return {"keys": [], "overlap_rows": 0, "examples": []}
    ktrain = train[keys].drop_duplicates(); ktest = test[keys].drop_duplicates()
    merged = ktrain.merge(ktest, on=keys, how="inner"); ex = merged.head(5).to_dict(orient="records")
    return {"keys": keys, "overlap_rows": int(len(merged)), "examples": ex}

def _constant_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

def _near_duplicate_cols(df: pd.DataFrame, sample: int = 20000, threshold: float = 0.999) -> List[Tuple[str,str]]:
    cols = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    if len(cols) < 2: return []
    n = min(sample, len(df)); idx = np.random.default_rng(0).choice(len(df), size=n, replace=False)
    pairs = []
    for i in range(len(cols)):
        x = df[cols[i]].iloc[idx].values
        for j in range(i+1, len(cols)):
            y = df[cols[j]].iloc[idx].values
            if np.nanstd(x) == 0 or np.nanstd(y) == 0: continue
            corr = np.corrcoef(x, y)[0,1]
            if np.isfinite(corr) and abs(corr) >= threshold: pairs.append((cols[i], cols[j]))
    return pairs

def run_checks(cfg: dict) -> Dict[str, Any]:
    train = pd.read_csv(Path(cfg["data"]["train"]).resolve()); test = pd.read_csv(Path(cfg["data"]["test"]).resolve())
    target = cfg["data"].get("target"); id_col = cfg["data"].get("id_col"); dt_col = cfg["data"].get("datetime_col")
    report: Dict[str, Any] = {"summary": {}, "leak": {}, "quality": {}, "hints": []}
    report["quality"]["n_train"] = int(len(train)); report["quality"]["n_test"] = int(len(test))
    report["quality"]["train_cols"] = train.columns.tolist(); report["quality"]["test_cols"] = test.columns.tolist()
    report["quality"]["missing_train_top"] = train.isna().mean().sort_values(ascending=False).head(20).to_dict()
    report["quality"]["missing_test_top"] = test.isna().mean().sort_values(ascending=False).head(20).to_dict()
    report["quality"]["constant_cols_train"] = _constant_cols(train.drop(columns=[c for c in [target] if c and c in train.columns]))
    report["quality"]["constant_cols_test"] = _constant_cols(test)
    report["quality"]["near_duplicate_pairs_train"] = _near_duplicate_cols(train)
    leak: Dict[str, Any] = {}
    leak["target_in_test"] = bool(target and target in test.columns)
    if leak["target_in_test"]: report["hints"].append(f"Target '{target}' exists in test.csv.")
    if target and target in train.columns:
        y = train[target]; num_cols = [c for c in train.columns if c != target and train[c].dtype.kind in "biufc"]
        high_corr = []; high_auc = []
        for c in num_cols:
            corr = _corr_numeric(y, train[c]); 
            if corr is not None and abs(corr) >= 0.98: high_corr.append({"col": c, "corr": corr})
            auc = _safe_auc(y, train[c])
            if auc is not None and auc >= 0.99: high_auc.append({"col": c, "auc": auc})
        leak["high_corr_with_target"] = high_corr; leak["single_feature_auc_ge_0.99"] = high_auc
    keys = [k for k in [id_col, dt_col] if k]; leak["train_test_key_overlap"] = _leak_by_dup_keys(train, test, keys)
    dup_rows = int(train.duplicated().sum()); leak["duplicate_rows_in_train"] = dup_rows
    if dup_rows > 0: report["hints"].append(f"{dup_rows} duplicate rows in train.")
    report["leak"] = leak
    flags = []
    if leak.get("target_in_test"): flags.append("target_in_test")
    if leak.get("high_corr_with_target"): flags.append("high_corr")
    if leak.get("single_feature_auc_ge_0.99"): flags.append("single_feature_auc")
    if leak["train_test_key_overlap"].get("overlap_rows", 0) > 0: flags.append("key_overlap")
    if dup_rows > 0: flags.append("dup_rows")
    report["summary"]["flags"] = flags; report["summary"]["ok"] = len(flags) == 0
    return report
