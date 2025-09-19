
from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
from src.utils import ROOT
from .features import add_time_features

def run_inference(cfg: dict, run_id: str):
    model_path = ROOT / cfg["output"]["dir"] / "models" / run_id / "model.joblib"
    pack = joblib.load(model_path)
    est = pack["est"]; cols = pack["cols"]
    target = pack["target"]; dt_col = pack["dt_col"]; id_col = pack["id_col"]

    test = pd.read_csv(Path(cfg["data"]["test"]).resolve())
    test = add_time_features(test, dt_col)

    history_path = cfg["data"].get("history")
    if history_path:
        hist = pd.read_csv(Path(history_path).resolve())
        use_cols = [c for c in [id_col, dt_col, target] if c and c in hist.columns]
        hist = hist[use_cols].copy()
        df = pd.concat([hist, test], axis=0, ignore_index=True)
        df = df.sort_values(([id_col] if id_col and id_col in df.columns else []) + [dt_col])
        for L in (1,7,14):
            df[f"lag_{L}"] = df[target].shift(L)
        for W in (7,14):
            df[f"rmean_{W}"] = df[target].shift(1).rolling(W).mean()
            df[f"rstd_{W}"] = df[target].shift(1).rolling(W).std()
        test = df[df[dt_col].isin(pd.to_datetime(test[dt_col]))].copy()

    X = test[[c for c in cols if c in test.columns]].copy()
    pred = est.predict(X)
    label = cfg["output"].get("label", "prediction")

    sub = pd.DataFrame({label: pred})
    id_col_cfg = cfg["data"].get("id_col")
    if id_col_cfg and id_col_cfg in test.columns: sub.insert(0, id_col_cfg, test[id_col_cfg].values)
    out_dir = ROOT / cfg["output"]["dir"] / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"; sub.to_csv(out_path, index=False); return out_path
