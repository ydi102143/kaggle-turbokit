from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils import ROOT

def _search_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1", step: float = 0.001):
    assert metric in ("f1", "acc")
    best_t, best_s = 0.5, -1.0
    thr = np.arange(0.0, 1.0+1e-12, step)
    for t in thr:
        pred = (proba >= t).astype(int)
        if metric == "f1":
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            p = 0.0 if (tp+fp)==0 else tp/(tp+fp)
            r = 0.0 if (tp+fn)==0 else tp/(tp+fn)
            s = 0.0 if (p+r)==0 else 2*p*r/(p+r)
        else:
            s = (pred == y_true).mean()
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

def optimize_threshold(cfg: dict, run_id: str, metric: str = "f1", step: float = 0.001, out_name: str | None = None):
    out_dir = ROOT / cfg["output"]["dir"]
    oof_path = out_dir / "oof" / f"{run_id}.csv"
    pred_path = out_dir / "preds" / f"{run_id}.csv"
    assert oof_path.exists(), f"OOF not found: {oof_path}"
    assert pred_path.exists(), f"Pred not found: {pred_path}"

    oof = pd.read_csv(oof_path)
    assert "oof_prob" in oof.columns, "oof_prob is required for threshold optimization (binary)."
    y_true = (oof["y"].values).astype(int)
    proba = oof["oof_prob"].values.astype(float)
    t, score = _search_threshold(y_true, proba, metric=metric, step=step)

    sub = pd.read_csv(pred_path)
    label = sub.columns[-1]
    if sub[label].between(0.0, 1.0).all():
        sub[label] = (sub[label].values >= t).astype(int)

    out_csv = out_dir / "preds" / (out_name or f"{run_id}_thres_{metric}.csv")
    sub.to_csv(out_csv, index=False)
    (out_csv.with_suffix(".meta.txt")).write_text(f"best_threshold={t}\nscore={score}\nmetric={metric}\n", encoding="utf-8")
    return str(out_csv)
