# Member C: CIFAR-10 and Class Imbalance

This branch adapts Team 69's shared CNN to ten classes and compares two ways of handling a long-tail training split: inverse-frequency loss weights and inverse-frequency resampling.

## Reproduce the experiments

Follow the [common environment setup](https://github.com/scy040320/ee6483-project2/tree/main) and switch to this branch:

```powershell
git switch c/cifar10
```

In `src/train_cifar10.py`, set `EXPERIMENT_MODE` in `main()` to one of `baseline`, `imbalanced`, `weighted_loss`, or `resampling`. From the repository root, run the following for each setting:

```powershell
.\.venv\Scripts\python.exe -m src.train_cifar10
```

All four modes use seed 42, batch size 64 and 20 epochs with Adam at learning rate 0.001. The script downloads CIFAR-10 to `data/raw`, splits its training set into 40,000 training and 10,000 validation images, and selects the checkpoint using validation accuracy. Each run reports accuracy on the official 10,000-image test split.

| Mode | Test accuracy |
| --- | ---: |
| Balanced baseline | 73.67% |
| Long tail, no correction | 43.36% |
| Inverse-frequency weighted loss | 53.44% |
| Inverse-frequency resampling | 56.69% |

After training the imbalanced and resampling modes, install the plotting dependencies and regenerate their confusion matrices:

```powershell
.\.venv\Scripts\python.exe -m pip install seaborn scikit-learn
.\.venv\Scripts\python.exe -m src.plot_cifar10_cm
```

The script writes `cm_imbalanced.png` and `cm_resampling.png` in the repository root. Preserve the existing run folders and plots before rerunning those experiments.

The recorded results are in [the experiment record](src/experiment_results.md). The committed confusion matrices cover [uncorrected long-tail training](cm_imbalanced.png) and [resampling](cm_resampling.png).
