# Member A: Dogs vs. Cats Baseline

This branch documents Team 69's data preparation, CNN baseline, parameter comparison and validation error analysis. See [main](https://github.com/scy040320/ee6483-project2/tree/main) for the reusable modules and common environment.

## Environment

Follow the setup on `main`, then switch to `a/baseline` and run the commands from the repository root. The original run used Python 3.11.9, PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130 on an NVIDIA GeForce RTX 5060 Laptop GPU.

```powershell
git switch a/baseline
$baselinePython = ".\.venv\Scripts\python.exe"
```

## Reproduce the experiments

Obtain the dataset supplied with the course and extract it under the repository root:

```text
datasets/
  train/cat/       10,000 images
  train/dog/       10,000 images
  val/cat/          2,500 images
  val/dog/          2,500 images
  test/               500 unlabelled images
```

The shared manifest, `data/manifests/cats_dogs_seed42.json`, records relative paths and labels for the course train and validation splits. Labels are cat = 0 and dog = 1. The baseline uses 128 × 128 RGB inputs and ImageNet channel normalization. Training augmentation uses a horizontal flip with probability 0.5 and rotation within ±15°, as specified in `configs/augmentation.json`. Validation uses deterministic preprocessing.

```powershell
& $baselinePython -m src.train --data-dir datasets --manifest data/manifests/cats_dogs_seed42.json --output-dir runs/baseline_reproduction --epochs 20 --image-size 128 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --dropout 0.3 --seed 42 --augment --augmentation-config configs/augmentation.json
& $baselinePython -m src.evaluate --data-dir datasets --checkpoint runs/baseline_reproduction/best_model.pt --output-dir runs/baseline_reproduction/evaluation
```

Training saves the settings, sample manifest, epoch history, learning curves, best checkpoint and final checkpoint. Evaluation reloads the best checkpoint and its manifest, then exports validation metrics, predictions, a confusion matrix and example images. Choose fresh output directories for repeat runs. Add `--device cpu` to run explicitly on CPU, or `--show-plots` to open the plots after saving them.

For the parameter comparison, keep the same settings and use separate run directories. Change only `--lr` to `0.0003` for the lower learning rate, or only `--batch-size` to `64` for the larger batch.

| Run | Best epoch | Validation accuracy |
| --- | ---: | ---: |
| Reference | 18 | 83.34% |
| Learning rate 0.0003 | 19 | 80.00% |
| Batch size 64 | 19 | 81.84% |

The shared modules are `src/data.py` and `src/augmentation.py` for loading and preprocessing, `src/models.py` for the configurable CNN, and `src/engine.py` for the model-independent training and evaluation loops. Use `python -m src.inspect_data --help` and `python -m src.preview_augmentation --help` for the data checks and preview options.

## Validation error analysis

The selected reference checkpoint correctly classified 4,167 of 5,000 validation images. With rows as true classes and columns as predicted classes, the confusion matrix is `[[2120, 380], [453, 2047]]` in cat, dog order. Evaluation exports per-image predictions and examples of correct and incorrect classifications. These scores concern the labelled validation split; the unlabelled course test set is handled on the transfer-learning branch.
