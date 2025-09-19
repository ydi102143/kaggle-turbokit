
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
from src.utils import ROOT

def run_stack(files: list[str], task: str = "regression", label: str | None = None, out_name: str | None = None):
    dfs = [pd.read_csv(Path(f)) for f in files]
    base = dfs[0].copy()
    if label is None:
        if base.shape[1] == 1:
            label = base.columns[0]; ids = None
        else:
            ids = base.columns[0]; label = base.columns[-1]
    else:
        ids = base.columns[0] if base.shape[1] > 1 else None
    X = np.column_stack([df[label].values for df in dfs])
    if task == "regression":
        pred = X.mean(axis=1)
    else:
        pred = X.mean(axis=1)
    out = base[[ids]].copy() if (base.shape[1] > 1 and ids) else pd.DataFrame()
    out[label] = pred
    out_dir = ROOT / "outputs" / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (out_name or f"stack_{len(files)}.csv")
    out.to_csv(path, index=False); return path
