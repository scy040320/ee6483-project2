"""PNG plots viewable directly in VS Code."""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_history(history, output, show=False):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                      marker="o", label=split)
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=metric.capitalize())
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def plot_confusion(predictions, class_names, output, show=False):
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for row in predictions:
        matrix[row["label"], row["prediction"]] += 1
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names,
             xlabel="Predicted label", ylabel="True label", title="Validation confusion matrix")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center",
                  color="white" if value > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def plot_examples(predictions, data_dir, class_names, output, correct=True,
                  seed=42, limit=4, show=False):
    candidates = [row for row in predictions
                  if (row["label"] == row["prediction"]) == correct]
    rows = random.Random(seed).sample(candidates, min(limit, len(candidates)))
    columns = 2
    figure, axes = plt.subplots(max(1, math.ceil(len(rows) / columns)), columns,
                                figsize=(9, 4 * max(1, math.ceil(len(rows) / columns))),
                                squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, rows):
        with Image.open(Path(data_dir) / row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{Path(row['path']).name}\nTrue: {class_names[row['label']]} | "
                       f"Predicted: {class_names[row['prediction']]}\n"
                       f"Confidence: {row['confidence']:.3f}")
    if not rows:
        axes[0, 0].text(0.5, 0.5, "No matching validation examples", ha="center")
    figure.suptitle("Correct validation examples" if correct else "Incorrect validation examples")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
