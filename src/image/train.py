
from __future__ import annotations
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm
from src.utils import ROOT, save_json, seed_everything
from src.logger import RunLogger
from .dataset import ImageCSVDataset

def run_training(cfg: dict, run_id: str):
    seed_everything(cfg.get("cv", {}).get("random_state", 42))
    run_logger = RunLogger(run_id, ROOT / cfg['output']['dir'])

    train_csv = Path(cfg["data"]["train"]).resolve()
    image_root = Path(cfg["data"].get("image_root", train_csv.parent))
    path_col = cfg["data"].get("path_col", "image_path")
    label_col = cfg["data"].get("label_col", "label")

    model_name = cfg["model"].get("name", "resnet18")
    batch_size = int(cfg["train"].get("batch_size", 32))
    epochs = int(cfg["train"].get("epochs", 3))
    lr = float(cfg["train"].get("lr", 1e-3))
    img_size = int(cfg["train"].get("img_size", 224))
    val_ratio = float(cfg["cv"].get("val_ratio", 0.2))

    tfm_train = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tfm_val = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    ds_full = ImageCSVDataset(str(train_csv), image_root=str(image_root), label_col=label_col, path_col=path_col, transform=tfm_train)
    num_classes = len(ds_full.class_to_idx)
    val_len = int(len(ds_full) * val_ratio); train_len = len(ds_full) - val_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len]); ds_val.dataset.transform = tfm_val

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_acc = 0.0
    best_path = ROOT / cfg["output"]["dir"] / "models" / run_id / "model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        net.train(); loss_sum = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); logits = net(xb); loss = criterion(logits, yb); loss.backward(); optimizer.step()
            loss_sum += loss.item() * xb.size(0)
        train_loss = loss_sum / len(ds_train)
        net.eval(); acc_sum = 0.0; cnt = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                acc_sum += (torch.argmax(logits, dim=1) == yb).float().sum().item()
                cnt += xb.size(0)
        val_acc = acc_sum / max(1, cnt)
        run_logger.log(step=epoch+1, train_loss=float(train_loss), val_acc=float(val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_name": model_name, "state_dict": net.state_dict(), "num_classes": num_classes, "class_to_idx": ds_full.class_to_idx}, best_path)
    save_json({"task": "image", "run_id": run_id, "model": model_name, "best_val_acc": best_acc, "img_size": img_size, "label_col": label_col, "path_col": path_col}, best_path.parent / "meta.json")
