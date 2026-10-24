"""Folder data and fixed manifests shared by members A and B."""

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common import seed_worker
from .augmentation import MEAN, STD, build_image_transform

CLASS_NAMES = ("cat", "dog")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_manifest(data_dir, seed=42, max_train_per_class=None, max_val_per_class=None,
                  class_names=CLASS_NAMES):
    """Use the supplied split; sample only within each existing class folder."""
    root = Path(data_dir).resolve()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    splits = {}
    for split, limit in (("train", max_train_per_class), ("val", max_val_per_class)):
        if limit is not None and limit < 1:
            raise ValueError("Per-class limits must be positive.")
        records = []
        for name, label in class_to_idx.items():
            folder = root / split / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class directory: {folder}")
            paths = sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not paths:
                raise ValueError(f"No images in {folder}")
            if limit is not None:
                if limit > len(paths):
                    raise ValueError(f"Requested {limit} images but {folder} has {len(paths)}")
                rng = random.Random(f"{seed}:{split}:{name}")
                paths = sorted(rng.sample(paths, limit))
            records.extend({"path": p.relative_to(root).as_posix(), "label": label}
                           for p in paths)
        splits[split] = records
    return {"version": 1, "seed": seed, "class_to_idx": class_to_idx, "splits": splits}


def load_manifest(path, data_dir, class_names=CLASS_NAMES):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {name: i for i, name in enumerate(class_names)}
    if manifest["class_to_idx"] != expected:
        raise ValueError(f"Manifest label mapping must be {expected}")
    root = Path(data_dir).resolve()
    seen = set()
    for split in ("train", "val"):
        records = manifest["splits"][split]
        if not records:
            raise ValueError(f"Empty manifest split: {split}")
        for record in records:
            relative = Path(record["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):
                raise ValueError("Manifest paths must stay inside the data directory")
            if record["label"] not in expected.values():
                raise ValueError("Invalid label in manifest")
            name = class_names[record["label"]]
            if not path.is_relative_to(root / split / name):
                raise ValueError(f"Split or label disagrees with path: {record['path']}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in seen:
                raise ValueError(f"Duplicate manifest image: {path}")
            seen.add(path)
    return manifest


def manifest_hash(manifest):
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_transforms(image_size=128, augment=False, augmentation_settings=None):
    steps = [build_image_transform(image_size, augment, augmentation_settings)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    def __init__(self, data_dir, records, transform):
        self.root = Path(data_dir).resolve()
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["path"]
        try:
            with Image.open(path) as image:
                inputs = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Cannot decode image: {path}") from exc
        return inputs, record["label"], record["path"]


def build_loaders(data_dir, batch_size=32, image_size=128, seed=42, augment=False,
                  num_workers=0, manifest_path=None, max_train_per_class=None,
                  max_val_per_class=None, class_names=CLASS_NAMES, pin_memory=False,
                  augmentation_settings=None):
    """Return ({'train': loader, 'val': loader}, manifest).

    Each batch is (images, integer labels, relative paths). Manifest selection
    takes precedence over per-class limits, which must not be supplied with it.
    A and B pass the same augmentation_settings dictionary for model comparisons.
    Validation always disables augmentation, regardless of the training flag.
    """
    if manifest_path and (max_train_per_class is not None or max_val_per_class is not None):
        raise ValueError("Use either --manifest or per-class limits, not both.")
    manifest = (load_manifest(manifest_path, data_dir, class_names) if manifest_path
                else make_manifest(data_dir, seed, max_train_per_class,
                                   max_val_per_class, class_names))
    loaders = {}
    for split in ("train", "val"):
        dataset = ImageDataset(data_dir, manifest["splits"][split],
                               build_transforms(image_size, augment and split == "train",
                                                augmentation_settings))
        generator = torch.Generator().manual_seed(seed + (split == "val"))
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=seed_worker, generator=generator,
        )
    return loaders, manifest
