
from __future__ import annotations
from pathlib import Path
import os, json, time, random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dirs(base: Path):
    for p in [base / "models", base / "oof", base / "preds", base / "logs"]:
        p.mkdir(parents=True, exist_ok=True)

def make_run_id(prefix: str = "exp") -> str:
    return f"{prefix}_{int(time.time())}"

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
