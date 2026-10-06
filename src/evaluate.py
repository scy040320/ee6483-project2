"""Load a saved model, evaluate the SAME validation images, and export figures."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from .common import resolve_device, set_seed, write_json
from .data import build_loaders, manifest_hash
from .engine import evaluate
from .models import build_model
from .plots import plot_confusion, plot_examples


def main():
    parser = argparse.ArgumentParser(description="Validate saved baseline and plot examples")
    parser.add_argument("--checkpoint", default="runs/baseline/best_model.pt")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", help="Default: checkpoint directory/evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    # Load only checkpoints produced by this project or another trusted source.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    set_seed(config["seed"])
    device = resolve_device(args.device)
    loaders, manifest = build_loaders(
        args.data_dir, batch_size=config["batch_size"], image_size=config["image_size"],
        seed=config["seed"], num_workers=args.num_workers,
        manifest_path=checkpoint_path.parent / "manifest.json",
        pin_memory=device.type == "cuda",
    )
    if manifest_hash(manifest) != config["manifest_sha256"]:
        raise ValueError("The sample manifest differs from the one used for training")
    model = build_model(config["num_classes"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, predictions = evaluate(model, loaders["val"], nn.CrossEntropyLoss(), device,
                                    return_predictions=True)
    output = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output-dir for this evaluation.")
    output.mkdir(parents=True)
    write_json(output / "metrics.json", {**metrics, "checkpoint_epoch": checkpoint["epoch"],
               "manifest_sha256": config["manifest_sha256"]})
    class_names = [name for name, label in sorted(manifest["class_to_idx"].items(), key=lambda x: x[1])]
    columns = ["path", "label", "prediction", "confidence"] + [f"prob_{name}" for name in class_names]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            record = {key: row[key] for key in columns[:4]}
            record.update({f"prob_{name}": row["probabilities"][i] for i, name in enumerate(class_names)})
            writer.writerow(record)
    plot_confusion(predictions, class_names, output / "confusion_matrix.png", args.show_plots)
    for correct, filename in ((True, "correct_examples.png"), (False, "incorrect_examples.png")):
        plot_examples(predictions, args.data_dir, class_names, output / filename,
                      correct=correct, seed=config["seed"], show=args.show_plots)
    print(f"Validation loss: {metrics['loss']:.4f}; accuracy: {metrics['accuracy']:.4f}")
    print(f"Figures and predictions: {output.resolve()}")


if __name__ == "__main__":
    main()
