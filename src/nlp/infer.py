
from __future__ import annotations
from pathlib import Path
import json, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import ROOT

def run_inference(cfg: dict, run_id: str):
    model_dir = ROOT / cfg["output"]["dir"] / "models" / run_id
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir); model.eval()
    maps = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in maps["id2label"].items()}

    test_csv = Path(cfg["data"]["test"]).resolve()
    text_col = cfg["data"].get("text_col", "text"); label = cfg["output"].get("label", "prediction")
    df = pd.read_csv(test_csv)
    enc = tokenizer(df[text_col].astype(str).tolist(), truncation=True, padding=True, max_length=cfg["train"].get("max_len", 256), return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits; preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    pred_labels = [id2label[int(i)] for i in preds]

    sub = pd.DataFrame({label: pred_labels})
    if "Id" in df.columns: sub.insert(0, "Id", df["Id"].values)
    out_dir = ROOT / cfg["output"]["dir"] / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"; sub.to_csv(out_path, index=False); return out_path
