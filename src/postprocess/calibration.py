from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from ..utils import ROOT

def _fit_platt(y_true: np.ndarray, p: np.ndarray):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(p.reshape(-1,1), y_true.astype(int))
    def f(x):
        import numpy as _np
        return lr.predict_proba(_np.asarray(x).reshape(-1,1))[:,1]
    return f

def _fit_isotonic(y_true: np.ndarray, p: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y_true.astype(int))
    def f(x):
        import numpy as _np
        return iso.predict(_np.asarray(x))
    return f

def fit_and_apply_calibration(cfg: dict, run_id: str, method: str = "platt"):
    out_dir = ROOT / cfg["output"]["dir"]
    oof_path = out_dir / "oof" / f"{run_id}.csv"
    pred_path = out_dir / "preds" / f"{run_id}.csv"
    oof = pd.read_csv(oof_path); sub = pd.read_csv(pred_path)
    assert "oof_prob" in oof.columns, "oof_prob is required for calibration (binary)."

    y_true = (oof["y"].values).astype(int)
    p = oof["oof_prob"].values.astype(float)
    if method == "platt":
        g = _fit_platt(y_true, p)
    elif method == "isotonic":
        g = _fit_isotonic(y_true, p)
    else:
        raise ValueError("method must be platt|isotonic")

    label = sub.columns[-1]
    if not sub[label].between(0.0, 1.0).all():
        return str(pred_path)
    sub[label] = g(sub[label].values)
    out = out_dir / "preds" / f"{run_id}_cal_{method}.csv"
    sub.to_csv(out, index=False)
    return str(out)
