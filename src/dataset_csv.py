import csv
import os
from typing import List, Tuple, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import random

class CSVImageDataset(Dataset):
    def __init__(self,
                 csv_file: Optional[str] = None,
                 images_root: Optional[str] = None,
                 transform: Optional[transforms.Compose] = None,
                 class_to_idx: Optional[Dict[str, int]] = None,
                 samples: Optional[List[Tuple[str, str]]] = None):
        self.csv_file = csv_file
        self.images_root = images_root
        self.transform = transform

        if samples is None:
            if not csv_file:
                raise ValueError("csv_file must be provided when samples is None")
            self.samples: List[Tuple[str, str]] = []
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if 'path' not in reader.fieldnames or 'label' not in reader.fieldnames:
                    raise ValueError("CSV must contain 'path' and 'label' headers")
                for row in reader:
                    img_path = row['path']
                    label = row['label']
                    if self.images_root and not os.path.isabs(img_path):
                        img_path = os.path.join(self.images_root, img_path)
                    self.samples.append((img_path, label))
        else:
            # Use provided samples directly
            self.samples = samples

        # Build or validate class mapping
        if class_to_idx is None:
            classes = sorted(list({lbl for _, lbl in self.samples}))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = self.class_to_idx[label_name]
        return img, target


def split_dataset(dataset: Dataset, val_split: float, seed: int = 42):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    return random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def stratified_indices_from_labels(labels: List[int], val_split: float, seed: int = 42):
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        by_class.setdefault(y, []).append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for y, idxs in by_class.items():
        n = len(idxs)
        n_val = max(1, int(round(n * val_split))) if n > 1 else 1
        rng.shuffle(idxs)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    # In case any class had only 1 sample and went entirely to val, ensure at least one train sample overall
    if not train_idx and val_idx:
        train_idx.append(val_idx.pop())
    return train_idx, val_idx


def make_csv_train_val_datasets(csv_file: str,
                                images_root: Optional[str],
                                train_tf,
                                val_tf,
                                val_split: float,
                                seed: int = 42):
    # Build full dataset once (no transform yet)
    full = CSVImageDataset(csv_file=csv_file, images_root=images_root, transform=None)
    class_to_idx = full.class_to_idx
    # derive numeric labels
    labels = [class_to_idx[lbl] for _, lbl in full.samples]
    train_idx, val_idx = stratified_indices_from_labels(labels, val_split, seed=seed)

    train_samples = [full.samples[i] for i in train_idx]
    val_samples = [full.samples[i] for i in val_idx]

    train_ds = CSVImageDataset(images_root=None, transform=train_tf, class_to_idx=class_to_idx, samples=train_samples)
    val_ds = CSVImageDataset(images_root=None, transform=val_tf, class_to_idx=class_to_idx, samples=val_samples)

    return train_ds, val_ds, class_to_idx
