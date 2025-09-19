from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .utils import ROOT

def _winsor(x: np.ndarray, p: float):
    if p <= 0: return x
    lo = np.quantile(x, p); hi = np.quantile(x, 1-p)
    return np.clip(x, lo, hi)

def _read_align(files):
    dfs = [pd.read_csv(f) for f in files]
    id_col = dfs[0].columns[0] if dfs[0].shape[1] > 1 else None
    label = dfs[0].columns[-1]
    if id_col is None:
        M = np.column_stack([d[label].values for d in dfs])
        return None, label, M, None
    base = dfs[0][[id_col, label]].rename(columns={label: "m0"})
    for i, d in enumerate(dfs[1:], start=1):
        base = base.merge(d[[id_col, label]].rename(columns={label: f"m{i}"}), on=id_col, how="inner")
    cols = [c for c in base.columns if c.startswith("m")]
    M = base[cols].values
    return id_col, label, M, base[[id_col]]

def run_blend_robust(files: list[str], method: str = "mean", winsor: float = 0.0):
    assert len(files) >= 2, "need >=2 files"
    id_col, label, M, ids = _read_align(files)
    M = np.column_stack([_winsor(M[:,i], winsor) for i in range(M.shape[1])])
    if method == "gmean":
        M = np.clip(M, 1e-12, None); blend = np.exp(np.mean(np.log(M), axis=1))
    elif method == "rank":
        from scipy.stats import rankdata
        R = np.column_stack([rankdata(M[:,i]) for i in range(M.shape[1])]); blend = R.mean(axis=1)
    else:
        blend = M.mean(axis=1)
    if ids is None:
        out = pd.DataFrame({label: blend})
    else:
        out = ids.copy(); out[label] = blend
    out_dir = ROOT / "outputs" / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"blend_robust_{len(files)}_{method}{'_w'+str(winsor) if winsor>0 else ''}.csv"
    out.to_csv(path, index=False); return path
