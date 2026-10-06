# EE6483 Project 2 — 小组公共分类流程

私有仓库：<https://github.com/scy040320/ee6483-project2>。公共流程已在 Python 3.11.9、PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130 和 RTX 5060 Laptop GPU 上完成图片读取、增强预览、小样本训练、验证及模型保存/加载检查。小样本结果用于验证流程，不作为正式实验结果。

## 1. 获取代码与安装环境

获授权成员接受仓库邀请后克隆，在 VS Code 中打开整个仓库文件夹：

```powershell
git clone https://github.com/scy040320/ee6483-project2.git
cd ee6483-project2
python -m venv .venv
```

在 VS Code 选择 `.venv/Scripts/python.exe` 作为解释器。以下是 Windows PowerShell 命令。

NVIDIA GPU：使用与设备和驱动兼容的 CUDA 构建。已验证的 CUDA 13.0 环境：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

仅使用 CPU：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` 记录我已运行成功的完整 CUDA 13.0 环境。需要匹配该环境时：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

CPU 环境使用上面的 CPU 安装命令，不直接安装包含 `+cu130` 的锁文件。其他平台与设备参考 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)。Windows 加载器默认 `num_workers=0`。

## 2. 数据与统一约定

课程数据由成员各自下载并放置，仓库不包含图片：

```text
datasets/
  train/cat/    10,000 张
  train/dog/    10,000 张
  val/cat/       2,500 张
  val/dog/       2,500 张
  test/            500 张，无标签
```

- 猫狗标签：`cat=0`、`dog=1`；主实验随机种子：`42`。
- 保持课程 train/val 划分，训练只更新训练集，验证集用于评估，测试集不参与调参。
- 全量共享清单：`data/manifests/cats_dogs_seed42.json`；A、B 按该文件使用同一批训练和验证样本。清单只有相对路径与标签，支持不同电脑的数据目录。
- 默认输入：RGB、128×128，归一化均值 `(0.485, 0.456, 0.406)`，标准差 `(0.229, 0.224, 0.225)`。
- 共享增强参数：`configs/augmentation.json`，水平翻转概率 0.5，随机旋转 ±15°；增强只用于训练，验证与测试只做固定预处理。
- 模型输出原始 logits，损失使用交叉熵；日志 accuracy 为 0 到 1 的小数。

增强在线生成，不修改原图和标签，不增加硬盘中的样本数量。模型对比统一数据、增强规则和其他实验条件；增强对比则明确改变增强开关。预训练模型如要求不同尺寸或预处理，应记录差异。

## 3. 运行公共流程

从仓库根目录使用 `python -m src.xxx`，不要直接运行 src 中的单个文件。

统计数据、预览原图与增强：

```powershell
.\.venv\Scripts\python.exe -m src.inspect_data --output-dir runs/data_summary --show-plots
.\.venv\Scripts\python.exe -m src.preview_augmentation --manifest data/manifests/cats_dogs_seed42.json --output-dir runs/augmentation_preview --show-plots
```

运行两轮小样本分类流程：

```powershell
.\.venv\Scripts\python.exe -m src.train --output-dir runs/quickstart --epochs 2 --max-train-per-class 100 --max-val-per-class 50 --seed 42 --augment --augmentation-config configs/augmentation.json --show-plots
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint runs/quickstart/best_model.pt --show-plots
```

训练使用 200 张训练图、100 张验证图。评估命令独立加载保存的最佳权重及同目录 `manifest.json`，按原验证样本重新计算指标。加载后的 accuracy 应与训练记录的 `best_val_accuracy` 一致。

生成文件：

```text
runs/quickstart/
  config.json              参数、预处理、设备与版本
  manifest.json            此次使用的固定样本清单
  history.csv              每轮 train/val loss、accuracy、学习率与耗时
  metrics.json             最佳验证准确率与轮次
  best_model.pt            最佳模型权重及配置
  last_model.pt            最近一轮权重及配置
  learning_curves.png
  evaluation/
    metrics.json
    predictions.csv
    confusion_matrix.png
    correct_examples.png
    incorrect_examples.png
```

PNG 可直接在 VS Code 中查看；`--show-plots` 打开图窗，关闭当前图窗后程序继续。重复运行需选择新的 `--output-dir`，程序不覆盖已有实验。完整图像解码检查可给 inspect_data 添加 `--verify-images`。

## 4. 可复用模块与接口

| 文件 | 职责 |
|---|---|
| src/data.py | 猫狗图片加载、固定样本清单、预处理、DataLoader |
| src/augmentation.py | 共享增强与参数读取 |
| src/models.py | 基础 CNN，可配置输出类别数 |
| src/engine.py | 与模型结构无关的训练和验证循环 |
| src/common.py | 随机种子、设备、JSON 配置及环境信息 |
| src/plots.py | 训练曲线、混淆矩阵与案例图 |
| src/train.py、src/evaluate.py | 基础 CNN 的训练和保存/加载入口 |
| src/inspect_data.py、src/preview_augmentation.py | 数据统计与预览入口 |

| 接口 | 返回 |
|---|---|
| build_loaders(data_dir, ..., manifest_path, augmentation_settings) | `({'train': loader, 'val': loader}, manifest)` |
| build_model(num_classes=2, dropout=0.3) | 模型，输出形状为 `N×num_classes` |
| train_one_epoch(model, loader, criterion, optimizer, device) | `{'loss': ..., 'accuracy': ..., 'samples': ...}` |
| evaluate(model, loader, criterion, device, return_predictions=False) | 指标；开启预测时返回 `(metrics, predictions)` |

训练循环支持 `(images, labels)` 和 `(images, labels, paths)` 两种 batch。当前 loss 汇总假定 criterion 返回平均损失。`return_predictions=True` 返回真实标签、预测、置信度、类别概率及图片路径（无路径的加载器返回空字符串）。

### B：复用猫狗数据与增强，替换模型

```python
from src.augmentation import load_augmentation_config
from src.common import resolve_device, set_seed
from src.data import build_loaders
from src.engine import train_one_epoch, evaluate

set_seed(42)
device = resolve_device("auto")
settings = load_augmentation_config("configs/augmentation.json")
loaders, manifest = build_loaders(
    "datasets", seed=42, image_size=128, batch_size=32,
    manifest_path="data/manifests/cats_dogs_seed42.json",
    augment=True, augmentation_settings=settings,
    pin_memory=(device.type == "cuda"),
)
```

在自己的入口中构建预训练模型及二分类输出层、损失和优化器，然后调用 `train_one_epoch`、`evaluate`。做无增强对照时设 `augment=False`，保留同一清单。若数据不在 datasets，修改 data_dir。

`src.train`、`src.evaluate` 当前只构建/重建 BaselineCNN；B 的预训练模型需要自己的训练与加载入口或扩展模型工厂，不能直接用该评估入口加载其他架构权重。保存时记录模型名称、结构参数、预处理、类别映射、权重和样本清单。测试集预测及 submission.csv 由 B 的入口实现。

### C：接入 CIFAR-10，复用模型和训练循环

```python
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.common import set_seed
from src.data import build_transforms
from src.models import build_model
from src.engine import train_one_epoch, evaluate

set_seed(42)
train_dataset = CIFAR10(
    root="data/raw", train=True, download=True,
    transform=build_transforms(image_size=32, augment=True),
)
test_dataset = CIFAR10(
    root="data/raw", train=False, download=True,
    transform=build_transforms(image_size=32, augment=False),
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
model = build_model(num_classes=10)
```

在自己的入口中将模型移到设备并配置优化器/损失，即可复用 engine。上例展示数据与模型接入；C 需从训练集另划验证集进行调参，最后才评估测试集，并实现自己的不平衡样本选择与至少两种处理方法。CIFAR-10 使用官方类别顺序，不套用猫狗标签或猫狗共享清单。

## 5. 协作规则

- B 在 `b/pretrained`、C 在 `c/cifar10` 分支开发，通过 Pull Request 合并，不直接修改 main。
- 公共接口、标签、清单或增强参数修改时同步三人。
- 每次运行保存配置、逐轮日志、最佳权重和设备信息。
- 数据、环境、模型权重、密钥和原始课程文件不提交；runs 为本地结果目录。需要共享的报告图表放入 report_assets。
- 不未经小组同意公开仓库或对外分享代码。
