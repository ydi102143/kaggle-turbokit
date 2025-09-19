
from __future__ import annotations
from src.train import run_training as tabular_train
from src.infer import run_inference as tabular_infer

def get_trainer(task: str):
    t = (task or "tabular").lower()
    if t in ("tabular", "classification", "regression"):
        return tabular_train
    if t in ("timeseries", "time"):
        from src.timeseries.train import run_training as ts_train; return ts_train
    if t in ("image",):
        from src.image.train import run_training as img_train; return img_train
    if t in ("nlp",):
        from src.nlp.train import run_training as nlp_train; return nlp_train
    raise ValueError(f"unsupported task: {task}")

def get_inferer(task: str):
    t = (task or "tabular").lower()
    if t in ("tabular", "classification", "regression"):
        return tabular_infer
    if t in ("timeseries", "time"):
        from src.timeseries.infer import run_inference as ts_infer; return ts_infer
    if t in ("image",):
        from src.image.infer import run_inference as img_infer; return img_infer
    if t in ("nlp",):
        from src.nlp.infer import run_inference as nlp_infer; return nlp_infer
    raise ValueError(f"unsupported task: {task}")
