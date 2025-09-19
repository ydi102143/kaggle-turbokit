
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import ROOT

def run_blend(files: list[str], method: str = "mean"):
    dfs = [pd.read_csv(Path(f)) for f in files]
    base = dfs[0].copy()
    if base.shape[1] == 1:
        label = base.columns[0]; ids = None
    else:
        ids = base.columns[0]; label = base.columns[-1]
    M = np.column_stack([df[label].values for df in dfs])
    if method == "gmean":
        M = np.clip(M, 1e-12, None); blend = np.exp(np.mean(np.log(M), axis=1))
    elif method == "rank":
        from scipy.stats import rankdata
        R = np.column_stack([rankdata(m) for m in M]); blend = R.mean(axis=1)
    else:
        blend = M.mean(axis=1)
    out = base[[ids]].copy() if ids else pd.DataFrame(); out[label] = blend
    out_dir = ROOT / "outputs" / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"blend_{len(files)}_{method}.csv"; out.to_csv(out_path, index=False); return out_path
