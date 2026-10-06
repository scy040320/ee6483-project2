"""Run from the project root with python -m src.train."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from .common import environment_info, resolve_device, set_seed, write_json
from .data import build_loaders, manifest_hash
from .engine import evaluate, train_one_epoch
from .models import build_model
from .plots import plot_history
from .augmentation import describe_preprocessing, load_augmentation_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train member A's baseline CNN")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="Enable train-only flip/rotation")
    parser.add_argument("--augmentation-config", help="Shared JSON settings; needs --augment")
    parser.add_argument("--manifest", help="Reuse a fixed sample list instead of scanning/sampling")
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-val-per-class", type=int)
    parser.add_argument("--show-plots", action="store_true", help="Open the curve window at the end")
    args = parser.parse_args()
    if args.augmentation_config and not args.augment:
        parser.error("--augmentation-config requires --augment")
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 8:
        parser.error("epochs/batch-size must be positive and image-size must be at least 8")
    if args.num_workers < 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid worker count, learning rate, weight decay, or dropout")
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output-dir to keep past runs.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    augmentation_settings = load_augmentation_config(args.augmentation_config)
    loaders, manifest = build_loaders(
        args.data_dir, batch_size=args.batch_size, image_size=args.image_size,
        seed=args.seed, augment=args.augment, num_workers=args.num_workers,
        manifest_path=args.manifest, max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class, pin_memory=device.type == "cuda",
        augmentation_settings=augmentation_settings,
    )
    output.mkdir(parents=True)
    config = vars(args).copy()
    config.update(model="baseline_cnn", num_classes=len(manifest["class_to_idx"]),
                  normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                  environment=environment_info(device), manifest_sha256=manifest_hash(manifest))
    config["preprocessing"] = {
        "train": describe_preprocessing(args.image_size, args.augment, augmentation_settings),
        "val": describe_preprocessing(args.image_size, False, augmentation_settings),
    }
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    model = build_model(config["num_classes"], args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_accuracy, best_epoch = -1.0, None
    print(f"Device: {device}; train: {len(loaders['train'].dataset)}; val: {len(loaders['val'].dataset)}")
    print(f"Classes: {manifest['class_to_idx']}; train augmentation: {args.augment}")
    if args.augment:
        print(f"Augmentation settings: {augmentation_settings}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Outputs: {output.resolve()}")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        training = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation = evaluate(model, loaders["val"], criterion, device)
        row = {"epoch": epoch, "train_loss": training["loss"],
               "train_accuracy": training["accuracy"], "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - start}
        history.append(row)
        improved = validation["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = validation["accuracy"], epoch
        checkpoint = {"model_state_dict": model.state_dict(), "config": config,
                      "epoch": epoch, "validation": validation,
                      "best_val_accuracy": best_accuracy, "best_epoch": best_epoch}
        torch.save(checkpoint, output / "last_model.pt")
        if improved:
            torch.save(checkpoint, output / "best_model.pt")
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, output / "learning_curves.png")
        write_json(output / "metrics.json", {"best_val_accuracy": best_accuracy,
                   "best_epoch": best_epoch, "completed_epochs": epoch,
                   "last_train": training, "last_val": validation})
        print(f"Epoch {epoch}/{args.epochs}: train loss={training['loss']:.4f}, "
              f"acc={training['accuracy']:.4f}; val loss={validation['loss']:.4f}, "
              f"acc={validation['accuracy']:.4f}; {row['seconds']:.1f}s")
    if args.show_plots:
        plot_history(history, output / "learning_curves.png", show=True)
    print(f"Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print("Next: python -m src.evaluate --checkpoint " + str(output / "best_model.pt"))


if __name__ == "__main__":
    main()
