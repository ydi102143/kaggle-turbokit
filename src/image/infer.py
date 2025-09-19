
from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
import timm
from src.utils import ROOT
from .dataset import ImageCSVDataset

def run_inference(cfg: dict, run_id: str):
    model_dir = ROOT / cfg["output"]["dir"] / "models" / run_id
    pack = torch.load(model_dir / "model.pt", map_location="cpu")
    model = timm.create_model(pack["model_name"], pretrained=False, num_classes=pack["num_classes"])
    model.load_state_dict(pack["state_dict"]); model.eval()
    idx_to_class = {v:k for k,v in pack["class_to_idx"].items()}

    test_csv = Path(cfg["data"]["test"]).resolve()
    image_root = Path(cfg["data"].get("image_root", test_csv.parent))
    path_col = cfg["data"].get("path_col", "image_path")
    label = cfg["output"].get("label", "prediction")
    img_size = int(cfg["train"].get("img_size", 224))

    tfm = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = ImageCSVDataset(str(test_csv), image_root=str(image_root), label_col=None, path_col=path_col, transform=tfm)

    preds, ids = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, sample_id = ds[i]
            logits = model(img.unsqueeze(0))
            pred_idx = int(torch.argmax(logits, dim=1).item())
            preds.append(idx_to_class[pred_idx]); ids.append(sample_id)

    sub = pd.DataFrame({label: preds})
    full = pd.read_csv(test_csv)
    if "Id" in full.columns:
        sub.insert(0, "Id", full["Id"].values)

    out_dir = ROOT / cfg["output"]["dir"] / "preds"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"; sub.to_csv(out_path, index=False); return out_path
