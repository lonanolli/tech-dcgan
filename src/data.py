import json
import os
import tarfile
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


def load_data(tgz_path: str, data_path: str = "data/flower_data") -> str:
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_path)

    img_folder = os.path.join(data_path, "jpg")
    return img_folder if os.path.exists(img_folder) else data_path


class FlowerDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str, transform: Optional[object] = None):
        with open(json_path, "r") as f:
            category_map = json.load(f)
        self.samples = [pic for imgs in category_map.values() for pic in imgs]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        flower_pic = self.samples[idx]
        img_path = os.path.join(self.img_dir, flower_pic)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
