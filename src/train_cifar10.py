import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

# 导入项目中可复用的模块
from src.common import set_seed, resolve_device
from src.data import build_transforms
from src.models import build_model
from src.engine import train_one_epoch, evaluate

def main():
    # 1. 基础设置：固定随机种子，自动获取设备 (GPU/CPU)
    set_seed(42)
    device = resolve_device("auto")
    print(f"当前使用的设备: {device}")

    # 2. 准备数据
    # CIFAR-10 图片默认尺寸较小，这里我们指定为 32x32，并应用同学A写好的预处理逻辑
    transform_train = build_transforms(image_size=32, augment=True)
    
    # 自动下载并加载 CIFAR-10 训练集
    print("正在加载 CIFAR-10 数据集...")
    full_train_dataset = CIFAR10(root="data/raw", train=True, download=True, transform=transform_train)
    
    # 根据要求，你需要从训练集中划出一部分作为验证集用于调参
    # CIFAR-10 训练集共 50000 张，这里我们分出 40000 张训练，10000 张验证
    train_size = 40000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 封装 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"数据加载完成！训练批次: {len(train_loader)}，验证批次: {len(val_loader)}")

    # 3. 构建模型、损失函数和优化器
    # 你的任务是十分类，所以 num_classes 必须指定为 10
    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率随便设了一个初始值，之后调参时可以改
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. 跑一轮训练作为测试
    print("\n--- 开始训练一轮测试 ---")
    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"训练一轮的结果: {train_metrics}")

    print("\n--- 开始验证测试 ---")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"验证的结果: {val_metrics}")
    
    print("\n恭喜！基础的 CIFAR-10 训练流程已成功跑通！")

if __name__ == "__main__":
    main()