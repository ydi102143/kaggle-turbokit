from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

@dataclass
class OOFResult:
    train_enc: pd.Series
    test_enc: pd.Series
    mapping: Dict[str, float]

def _smoothing(count, global_mean, cat_mean, m=5.0):
    return (count * cat_mean + m * global_mean) / (count + m)

def oof_target_encode(
    X_train: pd.Series,
    y: pd.Series,
    X_test: Optional[pd.Series],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task: str = "regression",
    m: float = 20.0,
) -> OOFResult:
    x = X_train.astype(str).fillna("__NA__")
    if task == "classification":
        classes = sorted(pd.unique(y))
        if len(classes) != 2:
            raise ValueError("oof_target_encode classification expects binary target")
        pos = classes[-1]
        y_bin = (y == pos).astype(float)
        global_mean = float(y_bin.mean())
    else:
        y_bin = y.astype(float)
        global_mean = float(y_bin.mean())

    oof = pd.Series(np.nan, index=x.index, dtype=float)
    for tr_idx, va_idx in folds:
        tr_x, tr_y = x.iloc[tr_idx], y_bin.iloc[tr_idx]
        stats = tr_x.groupby(tr_x).agg(["size"])
        means = tr_y.groupby(tr_x).mean()
        tmp = pd.concat([stats["size"], means], axis=1)
        tmp.columns = ["cnt", "mean"]
        tmp["enc"] = _smoothing(tmp["cnt"], global_mean, tmp["mean"], m=m)
        mapping = tmp["enc"].to_dict()
        oof.iloc[va_idx] = x.iloc[va_idx].map(mapping).fillna(global_mean).values

    stats_full = x.groupby(x).agg(["size"])
    means_full = y_bin.groupby(x).mean()
    tmp_full = pd.concat([stats_full["size"], means_full], axis=1)
    tmp_full.columns = ["cnt", "mean"]
    tmp_full["enc"] = _smoothing(tmp_full["cnt"], global_mean, tmp_full["mean"], m=m)
    mapping_full = tmp_full["enc"].to_dict()

    if X_test is not None:
        xt = X_test.astype(str).fillna("__NA__")
        test_enc = xt.map(mapping_full).fillna(global_mean)
    else:
        test_enc = pd.Series(dtype=float)

    return OOFResult(train_enc=oof, test_enc=test_enc, mapping=mapping_full)

def add_numeric_interactions(df: pd.DataFrame, cols: List[str], degree: int = 2, limit: int = 30) -> pd.DataFrame:
    import itertools
    cols = [c for c in cols if c in df.columns][:limit]
    out = df.copy()
    for a, b in itertools.combinations(cols, 2):
        out[f"{a}__x__{b}"] = df[a].astype(float) * df[b].astype(float)
    return out
