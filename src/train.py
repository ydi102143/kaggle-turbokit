
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline as SkPipe

from .utils import seed_everything, ensure_dirs, save_json, ROOT
from .validation import make_splits
from .preprocessing import build_preprocess
from .models import ModelRegistry
from .metrics import score_task
from .logger import RunLogger

def run_training(cfg: dict, run_id: str):
    logger = RunLogger(run_id, ROOT / cfg['output']['dir'])
    seed_everything(cfg.get("cv", {}).get("random_state", 42))

    train = pd.read_csv(Path(cfg["data"]["train"]).resolve())
    target = cfg["data"]["target"]; id_col = cfg["data"].get("id_col"); task = cfg.get("task", "regression")

    pre, *_ = build_preprocess(train, target, id_col)
    y = train[target].values
    X = train.drop(columns=[c for c in [target, id_col] if c in train.columns])

    splits = make_splits(task, y, method=cfg["cv"].get("method", "kfold"), folds=cfg["cv"].get("folds", 5),
                         shuffle=cfg["cv"].get("shuffle", True), random_state=cfg["cv"].get("random_state", 42))

    reg = ModelRegistry(task)
    est = reg.build(cfg["model"]["name"], cfg["model"].get("params", {}))

    oof_pred = np.zeros(len(train)); oof_prob = None
    if task == "classification": oof_prob = np.zeros(len(train))

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]; y_tr, y_va = y[tr_idx], y[va_idx]
        pipe = SkPipe([("pre", pre), ("est", est)]); pipe.fit(X_tr, y_tr)
        if task == "classification":
            if hasattr(pipe["est"], "predict_proba"):
                prob = pipe.predict_proba(X_va)[:,1]; pred = (prob >= 0.5).astype(int); oof_prob[va_idx] = prob
            else:
                pred = pipe.predict(X_va)
            oof_pred[va_idx] = pred
        else:
            pred = pipe.predict(X_va); oof_pred[va_idx] = pred
        sc = score_task(task, y_va, pred, prob if task == "classification" and 'prob' in locals() else None)
        logger.log(step=fold, **{f"val_{k}": v for k,v in sc.items() if v is not None})
        print(f"Fold {fold}: {sc}")

    scores = score_task(task, y, oof_pred, oof_prob if task == "classification" else None)
    logger.log(step=999999, **{f"OOF_{k}": v for k,v in scores.items() if v is not None})
    print(f"OOF: {scores}")

    full_pipe = SkPipe([("pre", pre), ("est", est)]); full_pipe.fit(X, y)

    out_dir = ROOT / cfg["output"]["dir"]; ensure_dirs(out_dir)
    model_dir = out_dir / "models" / run_id; model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipe, model_dir / "model.joblib")

    oof_dir = out_dir / "oof"; oof_dir.mkdir(parents=True, exist_ok=True)
    if task == "classification" and oof_prob is not None:
        pd.DataFrame({"oof_pred": oof_pred, "oof_prob": oof_prob, "y": y}).to_csv(oof_dir / f"{run_id}.csv", index=False)
    else:
        pd.DataFrame({"oof_pred": oof_pred, "y": y}).to_csv(oof_dir / f"{run_id}.csv", index=False)

    meta = {"task": task, "run_id": run_id, "model": cfg["model"]["name"], "scores": scores, "target": target, "id_col": id_col}
    save_json(meta, model_dir / "meta.json")
