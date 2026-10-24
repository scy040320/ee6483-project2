import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision.datasets import CIFAR10

from src.common import set_seed, resolve_device
from src.data import build_transforms
from src.models import build_model
from src.engine import train_one_epoch, evaluate

def get_imbalanced_indices(subset_indices, all_targets, num_classes=10, imb_ratio=0.01):
    class_indices = {i: [] for i in range(num_classes)}
    for idx in subset_indices:
        label = all_targets[idx]
        class_indices[label].append(idx)
        
    max_samples = max([len(indices) for indices in class_indices.values()])
    imbalanced_indices = []
    class_counts = []
    
    for c in range(num_classes):
        ratio = imb_ratio ** (c / (num_classes - 1.0))
        target_count = max(1, int(max_samples * ratio))
        truncated = class_indices[c][:target_count]
        imbalanced_indices.extend(truncated)
        class_counts.append(len(truncated))
        
    return imbalanced_indices, class_counts

def main():
    # EXPERIMENT CONFIGURATION
    EXPERIMENT_MODE = "resampling" 
    EPOCHS = 20
    
    set_seed(42)
    device = resolve_device("auto")
    output_dir = f"runs/cifar10_{EXPERIMENT_MODE}_ep{EPOCHS}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Running Mode: {EXPERIMENT_MODE} for {EPOCHS} Epochs ---")
    print(f"Results will be saved to: {output_dir}")

    transform_train = build_transforms(image_size=32, augment=True)
    transform_test = build_transforms(image_size=32, augment=False)
    
    full_train_dataset = CIFAR10(root="data/raw", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="data/raw", train=False, download=True, transform=transform_test)
    val_dataset_raw = CIFAR10(root="data/raw", train=True, download=True, transform=transform_test)
    
    train_size = 40000
    val_size = len(full_train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=generator
    )
    
    # 1. Prepare Data based on Mode
    if EXPERIMENT_MODE == "baseline":
        active_train_indices = train_indices
        class_counts = [len(train_indices) // 10] * 10
    else:
        active_train_indices, class_counts = get_imbalanced_indices(
            subset_indices=train_indices, 
            all_targets=full_train_dataset.targets, 
            num_classes=10, 
            imb_ratio=0.01
        )
        
    train_dataset = Subset(full_train_dataset, active_train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)

    # 2. Configure Sampler based on Mode
    if EXPERIMENT_MODE == "resampling":
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[full_train_dataset.targets[idx]] for idx in active_train_indices]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. Configure Loss based on Mode
    model = build_model(num_classes=10).to(device)
    if EXPERIMENT_MODE == "weighted_loss":
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        weights = weights / weights.sum() * 10.0
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc = 0.0
    
    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch}/{EPOCHS}] Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"🎯 Final Test Accuracy ({EXPERIMENT_MODE}): {test_metrics['accuracy']:.4f}\n")

if __name__ == "__main__":
    main()