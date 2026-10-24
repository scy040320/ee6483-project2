# EE6483 Team 69 — B 20-epoch update

Contributor: **ZHU YANGKE**. Experiment/report update: 24 October 2026.

The original report, original A/C code, old 10-epoch results and Git branches were not overwritten. This package is an additive B update; no merge to main was performed.

## Results

| Run | Model | Pixels | Augmentation | Completed | Selected epoch | Validation accuracy | Macro-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| cnn128_noaug | baseline_cnn | 128 | False | 20/20 | 18 | 86.26% | 0.8625 |
| cnn128_aug | baseline_cnn | 128 | True | 20/20 | 19 | 83.76% | 0.8374 |
| resnet128_noaug | resnet18 | 128 | False | 20/20 | 19 | 97.34% | 0.9734 |
| resnet128_aug | resnet18 | 128 | True | 20/20 | 8 | 97.38% | 0.9738 |
| resnet224_aug | resnet18 | 224 | True | 20/20 | 6 | 98.96% | 0.9896 |

Selected B candidate: `resnet224_aug`, epoch 6; validation accuracy **98.96%** on 5,000 labelled images; macro-F1 0.9896. The test set has no ground truth, so no test accuracy is claimed.

The four 128-pixel runs form the main model/augmentation comparison. The 224-pixel run is a separate resolution control. All use seed 42, the same 20,000/5,000 manifest, batch size 32 and 20 fresh epochs. ResNet18 uses ImageNet-1K V1 weights and layer4 + FC fine-tuning. Head/backbone rates are 0.001/0.0001; CNN rate is 0.001. All use AdamW, weight decay 0.0001, no scheduler. The old 10-epoch tuning/seed analysis is historical and is not a 20-epoch replication statistic.

## Files

- `Group69 mini project_B20_updated.docx`: editable English B additions in the original group template. A/C source content is retained; literature survey and C-module sections remain for the team.
- `Group69 mini project_B20_updated.pdf`: native Microsoft Word rendering with updated TOC/page numbers.
- `selected_submission/submission.csv`: new selected model predictions for IDs 1–500, integer labels cat=0/dog=1.
- `selected_submission/test_predictions.csv`: dog probabilities, separate from submission labels.
- `selected_submission/provenance.json`: exact checkpoint hash, manifest hash and selected epoch.
- Best-model checkpoints were independently reloaded and checked, but are excluded from Git under the repository rules. They are supplied in the separately shared `B20_report_and_evidence.zip`.
- `experiments/<run>/{config.json,manifest.json,history.csv,metrics.json}`: settings and all 20 epoch rows.
- `experiments/<run>/evaluation/`: per-image validation predictions, class metrics, confusion matrices and case grids.
- `comparison_20epochs.csv`, `analysis.json`, `figures/`: report evidence and aggregated plots.
- `src/`: unchanged snapshot of shared A APIs. `b_module/`: unchanged B implementation snapshot. Do not overwrite a teammate’s newer shared source with this snapshot.
- `reproduce_20epoch.py`: portable experiment runner; executes all five runs, evaluation, validation-only selection and test inference.
- `delivery_checks.json`: source report/code preservation and submission checks.

This Git directory excludes datasets, pretrained weights, all checkpoint binaries and the large transfer ZIP. The separately shared `B20_report_and_evidence.zip` contains the five best-model checkpoints. To evaluate without retraining, extract its `experiments/<run>/best_model.pt` files into the matching directories here. The local raw experiment directories retain final-epoch checkpoints.

## Run the code (recorded Linux/WSL environment)

Use Python 3.12 and the same package/hardware environment where practical. From this package root:

```bash
python3.12 -m venv .venv_b20
source .venv_b20/bin/activate
python -m pip install -r environment-lock.txt
python -m b_module.preflight --data-dir "DATA_ROOT" --sample-submission report_assets/sampleSubmission.csv --verify-images
python reproduce_20epoch.py --data-dir "DATA_ROOT" --output-dir "runs/b20_reproduction" --device cuda --num-workers 4
```

Replace `DATA_ROOT` with the extracted course folder containing `train/`, `val/` and `test/`. The dataset is not Kaggle’s original split; use the course Drive archive: https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view . Its SHA256 is recorded in `input_verification.json`. Output directories must be new; the runner refuses to overwrite existing results. First training use downloads official ResNet18 V1 weights. Check the supplied SHA256 in `pretrained_weight_verification.json`. Native Windows users need corresponding platform-specific PyTorch installation rather than Linux-only dependency wheels. Use `--device cpu` if necessary; runtime will be much longer.

After extracting the selected checkpoint from the separate transfer ZIP, evaluate without retraining:

```bash
python -m b_module.evaluate --checkpoint "experiments/resnet224_aug/best_model.pt" --data-dir "DATA_ROOT" --output-dir "runs/reloaded_validation" --device cuda --sensitivity-exclude val/cat/cat.12408.jpg
python -m b_module.predict --checkpoint "experiments/resnet224_aug/best_model.pt" --test-dir "DATA_ROOT/test" --sample-submission report_assets/sampleSubmission.csv --output-dir "runs/new_test_predictions" --device cuda
```

Full `config.json` retains the actual local execution paths for audit. The reproduction runner substitutes the paths you provide. Seeding and deterministic algorithms (warn-only) do not guarantee bitwise equality on different hardware/software.

## Handoff / integration

Review this new Word copy and its B additions under Sections 2–7. Retain A’s original 83.34% as its reported baseline; do not replace it with a new B control. Use the new five-run 20-epoch table for the matched comparison. The earlier 10-epoch submission is retained in the old delivery; use this package’s new submission only after the team agrees which model to submit. The joint literature review and C work remain the team’s responsibility. This directory is the additive Git publication on `b/pretrained`. It does not replace root-level A/C source, earlier B outputs, or the original group report. No merge into `main` is included. `src/` and `b_module/` inside this directory are frozen reproduction snapshots, not changes to the repository root implementations. Historical Git-ref and preservation records describe the experiment/report preparation stage before publication. `PUBLISHED_SHA256.json` records hashes of all files in this published directory except itself.
