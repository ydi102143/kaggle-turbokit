
from __future__ import annotations
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_squared_error

def score_task(task: str, y_true, y_pred, y_prob=None):
    if task == "classification":
        auc = roc_auc_score(y_true, y_prob) if (y_prob is not None) else None
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        return {"f1": float(f1), "acc": float(acc), "auc": None if auc is None else float(auc)}
    else:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return {"rmse": float(rmse)}
