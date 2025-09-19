
from __future__ import annotations
import optuna, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline as SkPipe
from .utils import ROOT, seed_everything, make_run_id, save_json
from .preprocessing import build_preprocess
from .validation import make_splits
from .metrics import score_task
from .models import ModelRegistry

def _suggest_params(model: str, trial: optuna.Trial):
    m = model.lower()
    if m == "lightgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 500, 2000),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 31, 255),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 60))
    if m == "xgboost":
        return dict(n_estimators=trial.suggest_int("n_estimators", 400, 1600),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0))
    if m == "catboost":
        return dict(depth=trial.suggest_int("depth", 4, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    iterations=trial.suggest_int("iterations", 600, 2000))
    return {}

def run_tuning(cfg: dict, n_trials: int = 30):
    task = cfg.get("task", "regression")
    if task == "nlp":
        return run_tuning_nlp(cfg, n_trials)
    seed_everything(cfg.get("cv", {}).get("random_state", 42))
    model = cfg["model"]["name"]
    train = pd.read_csv(Path(cfg["data"]["train"]).resolve())
    target = cfg["data"]["target"]; id_col = cfg["data"].get("id_col")
    pre, _, _ = build_preprocess(train, target, id_col)
    y = train[target].values
    X = train.drop(columns=[c for c in [target, id_col] if c in train.columns])
    splits = make_splits(task, y, method=cfg["cv"].get("method", "kfold"), folds=cfg["cv"].get("folds", 5),
                         shuffle=cfg["cv"].get("shuffle", True), random_state=cfg["cv"].get("random_state", 42))
    reg = ModelRegistry(task)

    def objective(trial: optuna.Trial):
        params = _suggest_params(model, trial)
        est = reg.build(model, params)
        oof = np.zeros(len(train)); oof_prob = None
        if task == "classification": oof_prob = np.zeros(len(train))
        for tr_idx, va_idx in splits:
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]; y_tr, y_va = y[tr_idx], y[va_idx]
            pipe = SkPipe([("pre", pre), ("est", est)]); pipe.fit(X_tr, y_tr)
            if task == "classification" and hasattr(pipe["est"], "predict_proba"):
                prob = pipe.predict_proba(X_va)[:,1]; pred = (prob >= 0.5).astype(int); oof_prob[va_idx] = prob
            else:
                pred = pipe.predict(X_va)
            oof[va_idx] = pred
        scores = score_task(task, y, oof, oof_prob if task == "classification" else None)
        return scores.get("f1", -scores.get("rmse", 1e9)) if task == "classification" else -scores.get("rmse", 1e9)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = _suggest_params(model, study.best_trial)
    est = reg.build(model, best_params); pipe = SkPipe([("pre", pre), ("est", est)]); pipe.fit(X, y)
    run_id = make_run_id(prefix=cfg.get("name", "exp_tuned"))
    out_dir = ROOT / cfg["output"]["dir"] / "models" / run_id; out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    save_json({"best_params": best_params, "best_value": float(study.best_value), "task": task, "model": model}, out_dir / "tuning.json")
    return run_id, best_params, float(study.best_value)

def run_tuning_nlp(cfg: dict, n_trials: int = 10):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    task = "nlp"
    model_name = cfg["model"].get("name", "distilbert-base-uncased")
    train_path = Path(cfg["data"]["train"]).resolve()
    text_col = cfg["data"].get("text_col", "text"); label_col = cfg["data"].get("label_col", "label")
    df = pd.read_csv(train_path)
    texts = df[text_col].astype(str).tolist(); labels = df[label_col].tolist()
    label2id = {l:i for i,l in enumerate(sorted(set(labels)))}; id2label = {i:l for l,i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(model_name)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42); tr_idx, va_idx = None, None
    for tr, va in skf.split(texts, labels): tr_idx, va_idx = tr, va
    tr_texts = [texts[i] for i in tr_idx]; tr_labels = [labels[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]; va_labels = [labels[i] for i in va_idx]

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tok, max_len): self.texts=texts; self.labels=labels; self.tok=tok; self.max_len=max_len
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            enc=self.tok(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len)
            import torch as T
            item={k: T.tensor(v) for k,v in enc.items()}; item["labels"]=T.tensor(label2id[self.labels[i]]); return item

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        bs = trial.suggest_categorical("batch_size", [8, 16, 32])
        max_len = trial.suggest_categorical("max_len", [128, 256])
        epochs = trial.suggest_int("epochs", 1, 3)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
        ds_tr = DS(tr_texts, tr_labels, tok, max_len); ds_va = DS(va_texts, va_labels, tok, max_len)
        args = TrainingArguments(output_dir="/tmp/hf_out", per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
                                 learning_rate=lr, num_train_epochs=epochs, evaluation_strategy="epoch", save_strategy="no", logging_steps=50)
        def compute_metrics(eval_pred):
            logits, labels_ids = eval_pred; import numpy as np
            preds = logits.argmax(axis=-1)
            return {"accuracy": accuracy_score(labels_ids, preds), "f1": f1_score(labels_ids, preds, average="macro")}
        trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, compute_metrics=compute_metrics)
        trainer.train(); eval_metrics = trainer.evaluate()
        return float(eval_metrics.get("f1", eval_metrics.get("accuracy", 0.0)))

    study = optuna.create_study(direction="maximize"); study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return "tuned_nlp", study.best_trial.params, float(study.best_value)
