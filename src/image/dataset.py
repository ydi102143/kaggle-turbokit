
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageCSVDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str = "", label_col: str = "label",
                 path_col: str = "image_path", transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.label_col = label_col
        self.path_col = path_col
        self.transform = transform
        if self.label_col in self.df.columns:
            classes = sorted(self.df[self.label_col].unique())
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
            self.idx_to_class = {i:c for c,i in self.class_to_idx.items()}
        else:
            self.class_to_idx = None; self.idx_to_class = None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / str(row[self.path_col])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        if self.label_col in self.df.columns:
            y = self.class_to_idx[row[self.label_col]]
            return img, y
        else:
            return img, row.get("Id", idx)
