# EE6483 Mini Project 2: Shared Classification Pipeline

This is the reusable code for Team 69's image classification project. The default branch provides the data loader, preprocessing, configurable CNN and shared training and evaluation loops. Each member's experiment instructions and results are documented on their own branch.

| Branch | Purpose |
| --- | --- |
| `main` | Shared dataset handling, augmentation, model building blocks and training tools |
| [`a/baseline`](https://github.com/scy040320/ee6483-project2/tree/a/baseline) | A: dogs-vs.-cats baseline, parameter comparison and error analysis |
| [`b/pretrained`](https://github.com/scy040320/ee6483-project2/tree/b/pretrained) | B: ResNet18 transfer learning, matched experiments and test submission |
| [`c/cifar10`](https://github.com/scy040320/ee6483-project2/tree/c/cifar10) | C: CIFAR-10 classification and class imbalance experiments |

## Set up the shared environment

Run commands from the repository root. The Windows baseline was run with Python 3.11.9. Use the matching CPU or CUDA build of PyTorch for your machine.

```powershell
git clone https://github.com/scy040320/ee6483-project2.git
cd ee6483-project2
python -m venv .venv
$baselinePython = ".\.venv\Scripts\python.exe"
```

For the recorded CUDA 13.0 baseline environment:

```powershell
& $baselinePython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

For CPU only:

```powershell
& $baselinePython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
& $baselinePython -m pip install -r requirements.txt
```

The lock file contains CUDA-specific wheels. For a different platform or accelerator, use the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and install the remaining project dependencies. The Windows data loader defaults to zero workers.

## Use the shared code

| Module | What it provides |
| --- | --- |
| `src/data.py` | Folder datasets, deterministic train/validation manifests and data loaders |
| `src/augmentation.py` | Configurable training transforms and deterministic evaluation preprocessing |
| `src/models.py` | A compact CNN with a configurable number of output logits |
| `src/engine.py` | Model-independent training and evaluation loops |
| `src/train.py`, `src/evaluate.py` | Training and checkpoint reload for the shared baseline CNN |
| `src/inspect_data.py`, `src/preview_augmentation.py` | Dataset inspection and augmentation previews |
| `src/common.py`, `src/plots.py` | Reproducibility utilities, metrics and plots |

The dogs-vs.-cats loader expects the course data under `datasets/`, with `train/cat/`, `train/dog/`, `val/cat/` and `val/dog/`. The optional unlabelled `test/` folder is used by the transfer-learning submission workflow. The shared manifest at `data/manifests/cats_dogs_seed42.json` fixes the train/validation samples and the cat = 0, dog = 1 label mapping. The augmentation configuration at `configs/augmentation.json` is applied during training only.

From the repository root, run a small integration check:

```powershell
& $baselinePython -m src.train --output-dir runs/quickstart --epochs 2 --max-train-per-class 100 --max-val-per-class 50 --seed 42 --augment --augmentation-config configs/augmentation.json
& $baselinePython -m src.evaluate --checkpoint runs/quickstart/best_model.pt
```

This check uses 200 training and 100 validation images. Choose a fresh output directory when rerunning it. For full experiments, use the commands and records on the corresponding branch.

`build_loaders` returns the training and validation loaders with the sample manifest. `build_model(num_classes=...)` returns raw logits for cross-entropy. The shared engine accepts batches of `(images, labels)` or `(images, labels, paths)`, so it can also be used with other models or datasets. The baseline checkpoint loader is specific to the shared CNN; use B's evaluation entry point for ResNet18.

Keep full course datasets, local environments, checkpoints and temporary run directories outside Git.
