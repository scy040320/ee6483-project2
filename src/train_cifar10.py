import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10

from src.common import set_seed, resolve_device
from src.data import build_transforms
from src.models import build_model
from src.engine import train_one_epoch, evaluate

def get_imbalanced_indices(subset_indices, all_targets, num_classes=10, imb_ratio=0.01):
    """
    Creates a long-tail imbalanced distribution by truncating indices.
    """
    # Group indices by class using a dictionary
    class_indices = {i: [] for i in range(num_classes)}
    for idx in subset_indices:
        label = all_targets[idx]
        class_indices[label].append(idx)
        
    # Find the maximum number of samples in any class
    max_samples = max([len(indices) for indices in class_indices.values()])
    
    imbalanced_indices = []
    distribution = {}
    
    # Calculate target counts using exponential decay and slice arrays
    for c in range(num_classes):
        ratio = imb_ratio ** (c / (num_classes - 1.0))
        target_count = max(1, int(max_samples * ratio)) # Ensure at least 1 sample
        
        truncated = class_indices[c][:target_count]
        imbalanced_indices.extend(truncated)
        distribution[c] = len(truncated)
        
    print("\n--- Imbalanced Training Data Distribution ---")
    for c, count in distribution.items():
        print(f"Class {c}: {count} samples")
    print("---------------------------------------------\n")
        
    return imbalanced_indices

def main():
    set_seed(42)
    device = resolve_device("auto")
    
    # Update output directory for the imbalanced experiment
    output_dir = "runs/cifar10_imbalanced"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using device: {device}. Results will be saved to: {output_dir}")

    transform_train = build_transforms(image_size=32, augment=True)
    transform_test = build_transforms(image_size=32, augment=False)
    
    print("Loading CIFAR-10 dataset...")
    full_train_dataset = CIFAR10(root="data/raw", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="data/raw", train=False, download=True, transform=transform_test)
    val_dataset_raw = CIFAR10(root="data/raw", train=True, download=True, transform=transform_test)
    
    train_size = 40000
    val_size = len(full_train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=generator
    )
    
    # Apply imbalance logic to the training indices
    # We set imb_ratio=0.01 (Class 9 will have 100x fewer samples than Class 0)
    imb_train_indices = get_imbalanced_indices(
        subset_indices=train_indices, 
        all_targets=full_train_dataset.targets, 
        num_classes=10, 
        imb_ratio=0.01
    )
    
    # Extract subsets using the new truncated indices
    train_dataset = Subset(full_train_dataset, imb_train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)

    # Wrap with DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_acc = 0.0
    
    print(f"\n--- Starting imbalanced training for {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Train -> Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val   -> Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best validation accuracy: {best_val_acc:.4f}. Model saved!")
            
    print("\n--- Training finished. Evaluating best model on test set ---")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"[Result] Final Test Accuracy (Imbalanced): {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()