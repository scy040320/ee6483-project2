"""Count original data, create a fixed manifest, and preview raw training images."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .common import write_json
from .data import CLASS_NAMES, IMAGE_SUFFIXES, make_manifest, manifest_hash


def main():
    parser = argparse.ArgumentParser(description="Inspect course data and create a shared manifest")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="runs/data_summary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-val-per-class", type=int)
    parser.add_argument("--verify-images", action="store_true", help="Decode/check ALL images; takes time")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.data_dir).resolve(), Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"{output} exists; choose another --output-dir")
    manifest = make_manifest(root, args.seed, args.max_train_per_class, args.max_val_per_class)
    counts, bad_images = {}, []
    folders = [f"{split}/{name}" for split in ("train", "val") for name in CLASS_NAMES] + ["test"]
    for folder in folders:
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        images = sorted(p for p in directory.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        counts[folder] = len(images)
        if args.verify_images:
            print(f"Checking {folder}: {len(images)} images")
            for path in images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.convert("RGB").load()
                except Exception as exc:
                    bad_images.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    selected = {split: {name: sum(row["label"] == label for row in records)
                        for label, name in enumerate(CLASS_NAMES)}
                for split, records in manifest["splits"].items()}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", {"all_image_counts": counts, "selected_counts": selected,
               "class_to_idx": manifest["class_to_idx"], "manifest_sha256": manifest_hash(manifest),
               "verified_images": args.verify_images, "bad_images": bad_images})
    figure, axis = plt.subplots(figsize=(9, 4))
    bars = axis.bar(list(counts), list(counts.values()))
    axis.bar_label(bars)
    axis.set(ylabel="Number of images", title="Original course dataset")
    figure.tight_layout()
    figure.savefig(output / "class_counts.png", dpi=160)
    if args.show_plots:
        plt.show()
    plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for label, name in enumerate(CLASS_NAMES):
        records = [row for row in manifest["splits"]["train"] if row["label"] == label][:4]
        for axis in axes[label]:
            axis.axis("off")
        for axis, row in zip(axes[label], records):
            with Image.open(root / row["path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{name}: {Path(row['path']).name}")
    figure.suptitle("Raw training examples (before preprocessing)")
    figure.tight_layout()
    figure.savefig(output / "data_preview.png", dpi=160)
    if args.show_plots:
        plt.show()
    plt.close(figure)
    print(f"Original counts: {counts}\nSelected counts: {selected}\nBad images: {len(bad_images)}")
    print(f"Shared manifest and figures: {output.resolve()}")
    if bad_images:
        raise RuntimeError("Corrupt images found. See summary.json; no images were removed.")


if __name__ == "__main__":
    main()
