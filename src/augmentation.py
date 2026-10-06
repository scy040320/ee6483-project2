"""Shared online augmentation: the preview and training use the same pipeline."""

import json
import math
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ROTATION_FILL = tuple(round(value * 255) for value in MEAN)
DEFAULT_AUGMENTATION = {"horizontal_flip_probability": 0.5, "rotation_degrees": 15.0}


def load_augmentation_config(path=None):
    """Read a shared JSON file or use explicit built-in defaults."""
    settings = (json.loads(Path(path).read_text(encoding="utf-8"))
                if path is not None else DEFAULT_AUGMENTATION.copy())
    return validate_augmentation_config(settings)


def validate_augmentation_config(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_AUGMENTATION):
        raise ValueError(f"Augmentation config needs exactly: {list(DEFAULT_AUGMENTATION)}")
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= settings["horizontal_flip_probability"] <= 1:
        raise ValueError("horizontal_flip_probability must be between 0 and 1")
    if not 0 <= settings["rotation_degrees"] <= 180:
        raise ValueError("rotation_degrees must be between 0 and 180")
    return dict(settings)


def build_image_transform(image_size=128, augment=False, settings=None):
    """RGB PIL image -> RGB PIL image, before tensor conversion/normalization.

    Random transforms are sampled afresh on every access; source files and
    labels are unchanged. Use augment=False for validation and test images.
    """
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    settings = validate_augmentation_config(settings if settings is not None else DEFAULT_AUGMENTATION)
    steps = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=settings["horizontal_flip_probability"]),
            transforms.RandomRotation(settings["rotation_degrees"],
                                      interpolation=InterpolationMode.BILINEAR,
                                      expand=False, fill=ROTATION_FILL),
        ]
    return transforms.Compose(steps)


def describe_preprocessing(image_size, augment, settings):
    """Record the effective settings, not just a flag or mutable file path."""
    settings = validate_augmentation_config(settings)
    return {
        "image_size": [image_size, image_size], "color_mode": "RGB",
        "resize_interpolation": "bilinear", "normalize_mean": list(MEAN),
        "normalize_std": list(STD), "augmentation_enabled": bool(augment),
        "augmentation": settings if augment else None,
        "rotation_interpolation": "bilinear" if augment else None,
        "rotation_fill_rgb": list(ROTATION_FILL) if augment else None,
        "rotation_expand": False if augment else None,
        "order": (["resize", "random_horizontal_flip", "random_rotation", "to_tensor", "normalize"]
                  if augment else ["resize", "to_tensor", "normalize"]),
    }
