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

def main():
    set_seed(42)
    device = resolve_device("auto")
    
    # Create output directory for this run
    output_dir = "runs/cifar10_baseline"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using device: {device}. Results will be saved to: {output_dir}")

    # Training set needs augmentation, validation/test sets do not
    transform_train = build_transforms(image_size=32, augment=True)
    transform_test = build_transforms(image_size=32, augment=False)
    
    # Download and load datasets
    print("Loading CIFAR-10 dataset...")
    full_train_dataset = CIFAR10(root="data/raw", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="data/raw", train=False, download=True, transform=transform_test)
    
    # Load again for validation set to apply test transforms (no augmentation)
    val_dataset_raw = CIFAR10(root="data/raw", train=True, download=True, transform=transform_test)
    
    train_size = 40000
    val_size = len(full_train_dataset) - train_size
    
    # Split indices
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=generator
    )
    
    # Extract subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)

    # Wrap with DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Build model, criterion, and optimizer
    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Full Training Loop ----------
    epochs = 10
    best_val_acc = 0.0
    
    print(f"\n--- Starting training for {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        # Train and evaluate
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Train -> Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val   -> Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save the best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best validation accuracy: {best_val_acc:.4f}. Model saved!")
            
    # ---------- Final Evaluation on Test Set ----------
    print("\n--- Training finished. Evaluating best model on test set ---")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"[Result] Final Test Accuracy (Baseline): {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()