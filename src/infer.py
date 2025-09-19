
from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
from .utils import ROOT

def run_inference(cfg: dict, run_id: str):
    model_path = ROOT / cfg["output"]["dir"] / "models" / run_id / "model.joblib"
    if not model_path.exists(): raise FileNotFoundError(f"model not found: {model_path}")
    pipe = joblib.load(model_path)

    test = pd.read_csv(Path(cfg["data"]["test"]).resolve())
    id_col = cfg["data"].get("id_col"); label = cfg["output"].get("label", "prediction")
    X_test = test.drop(columns=[c for c in [id_col] if c in test.columns])
    pred = pipe.predict(X_test)

    sub = pd.DataFrame({label: pred})
    if id_col and id_col in test.columns: sub.insert(0, id_col, test[id_col].values)

    out_dir = ROOT / cfg["output"]["dir"] / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"; sub.to_csv(out_path, index=False); return out_path
