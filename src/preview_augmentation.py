"""Preview augmentations on training images; never trains a model."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .augmentation import build_image_transform, describe_preprocessing, load_augmentation_config
from .common import set_seed, write_json
from .data import load_manifest, make_manifest


def main():
    parser = argparse.ArgumentParser(description="Show original vs online augmented training images")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--manifest", help="Optional shared manifest (only train records are sampled)")
    parser.add_argument("--output-dir", default="runs/augmentation_preview")
    parser.add_argument("--augmentation-config", default="configs/augmentation.json")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.variants <= 8:
        parser.error("--variants must be between 1 and 8")
    output, root = Path(args.output_dir), Path(args.data_dir).resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists. Choose a new --output-dir.")
    settings = load_augmentation_config(args.augmentation_config)
    base = build_image_transform(args.image_size, False, settings)
    augmented = build_image_transform(args.image_size, True, settings)
    manifest = (load_manifest(args.manifest, root) if args.manifest
                else make_manifest(root, seed=args.seed))
    rng = random.Random(args.seed)
    selected = []
    class_names = [name for name, label in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    for label, name in enumerate(class_names):
        candidates = [row for row in manifest["splits"]["train"] if row["label"] == label]
        if not candidates:
            raise ValueError(f"No training images for {name}")
        selected.append(rng.choice(candidates))
    set_seed(args.seed)
    figure, axes = plt.subplots(len(selected), args.variants + 1,
                                figsize=(3 * (args.variants + 1), 3 * len(selected)), squeeze=False)
    for row_index, record in enumerate(selected):
        with Image.open(root / record["path"]) as source:
            image = source.convert("RGB")
        for col_index, axis in enumerate(axes[row_index]):
            # Show PIL RGB images, not normalized tensors, so colors stay natural.
            axis.imshow(base(image) if col_index == 0 else augmented(image))
            axis.set_title("Resized original" if col_index == 0 else f"Augmented {col_index}")
            axis.set_xlabel(f"{class_names[record['label']]}: {Path(record['path']).name}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Training augmentation preview (labels unchanged)")
    figure.tight_layout()
    output.mkdir(parents=True)
    figure.savefig(output / "augmentation_preview.png", dpi=160)
    write_json(output / "preview_config.json", {
        "seed": args.seed, "selected_training_images": selected, "variants": args.variants,
        "preprocessing": describe_preprocessing(args.image_size, True, settings),
    })
    if args.show_plots:
        plt.show()
    plt.close(figure)
    print(f"Preview saved to: {(output / 'augmentation_preview.png').resolve()}")
    print("Images are transformed in memory; dataset files/counts are unchanged.")


if __name__ == "__main__":
    main()
