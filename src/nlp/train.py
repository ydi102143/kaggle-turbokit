
from __future__ import annotations
from pathlib import Path
import json, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.utils import ROOT, seed_everything
from src.logger import RunLogger

def run_training(cfg: dict, run_id: str):
    seed_everything(cfg.get("cv", {}).get("random_state", 42))
    run_logger = RunLogger(run_id, ROOT / cfg['output']['dir'])

    train_csv = Path(cfg["data"]["train"]).resolve()
    text_col = cfg["data"].get("text_col", "text"); label_col = cfg["data"].get("label_col", "label")
    model_name = cfg["model"].get("name", "distilbert-base-uncased")
    epochs = int(cfg["train"].get("epochs", 2)); batch_size = int(cfg["train"].get("batch_size", 16))
    lr = float(cfg["train"].get("lr", 2e-5)); max_len = int(cfg["train"].get("max_len", 256)); folds = int(cfg["cv"].get("folds", 5))

    df = pd.read_csv(train_csv)
    texts = df[text_col].astype(str).tolist(); labels = df[label_col].tolist()
    label2id = {l:i for i,l in enumerate(sorted(set(labels)))}; id2label = {i:l for l,i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42); tr_idx, va_idx = None, None
    for tr, va in skf.split(texts, labels): tr_idx, va_idx = tr, va
    tr_texts = [texts[i] for i in tr_idx]; tr_labels = [labels[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]; va_labels = [labels[i] for i in va_idx]

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tok, max_length):
            self.texts=texts; self.labels=labels; self.tok=tok; self.max_length=max_length
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            enc=self.tok(self.texts[i], truncation=True, padding="max_length", max_length=self.max_length)
            item={k: torch.tensor(v) for k,v in enc.items()}; item["labels"]=torch.tensor(label2id[self.labels[i]]); return item

    ds_tr = DS(tr_texts, tr_labels, tokenizer, max_len); ds_va = DS(va_texts, va_labels, tokenizer, max_len)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    out_dir = ROOT / cfg["output"]["dir"] / "models" / run_id; out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(output_dir=str(out_dir / "hf_outputs"), evaluation_strategy="epoch", save_strategy="epoch",
                             learning_rate=lr, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                             num_train_epochs=epochs, weight_decay=0.01, logging_steps=50, load_best_model_at_end=True,
                             metric_for_best_model="accuracy", save_total_limit=2)

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels_ids = eval_pred; preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels_ids, preds); f1 = f1_score(labels_ids, preds, average="macro")
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()

    model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f: json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    try:
        hist = trainer.state.log_history
        for i, rec in enumerate(hist):
            if isinstance(rec, dict):
                metrics = {k: float(v) for k, v in rec.items() if isinstance(v, (int, float))}
                if metrics: run_logger.log(step=i, **metrics)
    except Exception: pass
