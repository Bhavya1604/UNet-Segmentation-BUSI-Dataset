"""
BUSI (Breast Ultrasound Images) dataset loader with mask merging.

Use this module for a reproducible data pipeline outside the notebooks.
Mask merging: multiple masks per image are combined with logical OR
so that the model receives one binary ground-truth mask per sample.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def build_data_dict(data_dir: str) -> dict:
    """
    Scan BUSI folder structure and build a dict: unique_id -> {folder, image, masks}.

    data_dir: path to Dataset_BUSI_with_GT (contains benign/, malignant/, normal/).
    """
    data_dict = {}
    folders = ["benign", "malignant", "normal"]

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            start, end = fname.find("("), fname.find(")")
            if start == -1 or end == -1:
                continue
            image_id = fname[start + 1 : end]
            unique_id = f"{folder}_{image_id}"

            if unique_id not in data_dict:
                data_dict[unique_id] = {"folder": folder, "image": None, "masks": []}

            if "_mask" in fname:
                data_dict[unique_id]["masks"].append(fname)
            else:
                data_dict[unique_id]["image"] = fname

    return data_dict


class BUSIDataset(Dataset):
    """
    PyTorch Dataset for BUSI: loads image + merged binary mask per sample.

    - Images: grayscale, resized to (256, 256), normalized to [0, 1].
    - Masks: all mask files for that image merged with logical OR, then resized.
    - Optional augmentation: horizontal/vertical flip (set augment=True).
    """

    def __init__(self, data_dict: dict, data_dir: str, augment: bool = True):
        self.data_dict = data_dict
        self.data_dir = data_dir
        self.keys = list(data_dict.keys())
        self.augment = augment

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        import random

        key = self.keys[idx]
        sample = self.data_dict[key]
        folder = sample["folder"]
        image_name = sample["image"]
        mask_names = sample["masks"]

        img_path = os.path.join(self.data_dir, folder, image_name)
        image = Image.open(img_path).convert("L")
        image = image.resize((256, 256))
        image = np.array(image, dtype=np.float32) / 255.0

        final_mask = np.zeros((256, 256), dtype=np.uint8)
        for m in mask_names:
            mask_path = os.path.join(self.data_dir, folder, m)
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((256, 256))
            mask = (np.array(mask) > 0).astype(np.uint8)
            final_mask = np.logical_or(final_mask, mask).astype(np.uint8)

        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                final_mask = np.fliplr(final_mask).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                final_mask = np.flipud(final_mask).copy()

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        final_mask = torch.tensor(final_mask, dtype=torch.float32).unsqueeze(0)
        return image, final_mask
