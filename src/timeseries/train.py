
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor, LGBMClassifier
from src.utils import ROOT, seed_everything, ensure_dirs, save_json
from src.logger import RunLogger
from .features import add_time_features
from .validation import PurgedGroupTimeSeriesSplit

def run_training(cfg: dict, run_id: str):
    seed_everything(cfg.get("cv", {}).get("random_state", 42))
    logger = RunLogger(run_id, ROOT / cfg['output']['dir'])
    task = cfg.get("task", "timeseries")
    train = pd.read_csv(Path(cfg["data"]["train"]).resolve())

    target = cfg["data"]["target"]; id_col = cfg["data"].get("id_col")
    dt_col = cfg["data"].get("datetime_col", "date")

    df = add_time_features(train, dt_col)
    df = df.sort_values(([id_col] if id_col and id_col in df.columns else []) + [dt_col]).reset_index(drop=True)

    y = df[target].values
    drop_cols = [c for c in [target, id_col, dt_col] if c in df.columns]
    X = df.drop(columns=drop_cols)

    folds = cfg["cv"].get("folds", 5); embargo = int(cfg["cv"].get("embargo", 0))
    splitter = PurgedGroupTimeSeriesSplit(n_splits=folds, embargo=embargo)

    est = LGBMClassifier(**cfg["model"].get("params", {})) if task == "classification" else LGBMRegressor(**cfg["model"].get("params", {}))

    oof = np.zeros(len(df))
    for i, (tr_idx, va_idx) in enumerate(splitter.split(df, time_col=dt_col, group_col=id_col)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]; y_tr, y_va = y[tr_idx], y[va_idx]
        est.fit(X_tr, y_tr)
        if task == "classification" and hasattr(est, "predict_proba"):
            p = est.predict_proba(X_va)[:,1]; oof[va_idx] = p
        else:
            p = est.predict(X_va); oof[va_idx] = p
        # simple logging
        logger.log(step=i, val_rmse=float(np.sqrt(np.mean((p - y_va)**2))) if task!="classification" else None)

    est.fit(X, y)

    out_dir = ROOT / cfg["output"]["dir"]; ensure_dirs(out_dir)
    model_dir = out_dir / "models" / run_id; model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"est": est, "cols": X.columns.tolist(), "target": target, "dt_col": dt_col, "id_col": id_col}, model_dir / "model.joblib")
    save_json({"task": task, "run_id": run_id, "cols": X.columns.tolist(), "folds": folds, "embargo": embargo}, model_dir / "meta.json")
    pd.DataFrame({"y": y, "oof_pred": oof}).to_csv(out_dir / "oof" / f"{run_id}.csv", index=False)
