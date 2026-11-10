# Member B: ResNet18 Transfer Learning

This branch contains Team 69's pretrained ResNet18 implementation, controlled dogs-vs.-cats comparisons and test predictions. The shared pipeline is documented on [main](https://github.com/scy040320/ee6483-project2/tree/main).

## Final 20-epoch experiments

Start with the [20-epoch reproduction guide](b_updates/2026-10-24_20epochs/README.md). The package contains a frozen code snapshot, a Linux/WSL environment lock, a reproduction script, five training histories, validation predictions and report figures. Run the commands from inside the package directory.

| Configuration | Validation accuracy |
| --- | ---: |
| CNN, 128 × 128, no augmentation | 86.26% |
| CNN, 128 × 128, augmentation | 83.76% |
| ResNet18, 128 × 128, no augmentation | 97.34% |
| ResNet18, 128 × 128, augmentation | 97.38% |
| ResNet18, 224 × 224, augmentation | 98.96% |

The four 128 × 128 runs compare model and augmentation settings on the same split. The 224 × 224 run tests resolution separately. The final candidate is the 224 × 224 ResNet18 at epoch 6, selected using validation accuracy.

The [final submission CSV](b_updates/2026-10-24_20epochs/selected_submission/submission.csv) contains 500 `id,label` predictions, with cat = 0 and dog = 1. Probabilities and checkpoint provenance are in the same directory. The course test set has no ground-truth labels.

## Earlier experiments

The root-level [implementation](b_module/) and [results](B_RESULTS.md) document the earlier 10-epoch study. Use the 20-epoch package above for the final controlled comparison and submission. Checkpoint binaries are distributed separately from Git.
