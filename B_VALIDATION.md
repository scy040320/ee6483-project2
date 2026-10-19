# B 最终验证记录

全部结果来自2026-10-18本机实际运行。

```json
{
  "original_files_verified_unchanged": [
    ".gitignore",
    "README.md",
    "configs/augmentation.json",
    "data/manifests/cats_dogs_seed42.json",
    "requirements-lock.txt",
    "requirements.txt",
    "src/__init__.py",
    "src/augmentation.py",
    "src/common.py",
    "src/data.py",
    "src/engine.py",
    "src/evaluate.py",
    "src/inspect_data.py",
    "src/models.py",
    "src/plots.py",
    "src/preview_augmentation.py",
    "src/train.py"
  ],
  "formal_runs": [
    {
      "run": "b_a0_seed42",
      "best_val_accuracy": 0.795,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_a1_seed42",
      "best_val_accuracy": 0.7592,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_a1_seed43",
      "best_val_accuracy": 0.758,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_a1_seed44",
      "best_val_accuracy": 0.7446,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_b0_seed42",
      "best_val_accuracy": 0.9728,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_b1_seed42",
      "best_val_accuracy": 0.9738,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_b2_seed42",
      "best_val_accuracy": 0.9634,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_b3_seed42",
      "best_val_accuracy": 0.9722,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_b4_seed42",
      "best_val_accuracy": 0.9896,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_final_seed43",
      "best_val_accuracy": 0.9888,
      "completed_epochs": 10,
      "reload_verified": true
    },
    {
      "run": "b_final_seed44",
      "best_val_accuracy": 0.9886,
      "completed_epochs": 10,
      "reload_verified": true
    }
  ],
  "smoke_excluded": true,
  "image_integrity": {
    "checked": 25500,
    "bad_images": []
  },
  "final_checkpoint": "runs/b_b4_seed42/best_model.pt",
  "submission_rows": 500,
  "github": "not pushed; API access returned 404",
  "checks": [
    "GPU head/layer4 forward/backward and freeze tests",
    "all 25500 images decoded",
    "ID ordering / missing and duplicate detection / binary label tests",
    "independent reload of all 11 formal runs",
    "main and duplicate-excluded validation metrics",
    "all original files byte-identical"
  ]
}
```

交付ZIP包含最终模型checkpoint及各实验配置、日志和图。其他运行的中间/最近权重仍保存在完整工作副本，未重复放入ZIP。临时环境和解压数据位于/tmp，原始数据包持久保存在downloads。


交接包说明：本包的可提交素材位于report_assets/b_team69/；各实验轻量记录位于其experiments/子目录。模型和所选运行的本地结果位于02_LOCAL_RESULTS/runs/，复制到仓库根目录后才能按原runs路径加载。不要用完整旧仓库覆盖A/C的新版本。
