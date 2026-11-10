import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from src.common import resolve_device
from src.data import build_transforms
from src.models import build_model

def plot_confusion_matrix(model_path, title, output_filename):
    device = resolve_device("auto")
    transform_test = build_transforms(image_size=32, augment=False)
    test_dataset = CIFAR10(root="data/raw", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. Load your trained model
    model = build_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print(f"Generating predictions for {title}...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # 2. Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    # CIFAR-10 class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_filename}")

if __name__ == "__main__":
    # We will generate two plots to show the contrast!
    
    # Plot 1: The broken imbalanced model
    plot_confusion_matrix(
        model_path="runs/cifar10_imbalanced_ep20/best_model.pt", 
        title="CIFAR-10 Long-tail Imbalance (No Correction)", 
        output_filename="cm_imbalanced.png"
    )
    
    # Plot 2: The best corrected model (Resampling)
    plot_confusion_matrix(
        model_path="runs/cifar10_resampling_ep20/best_model.pt", 
        title="CIFAR-10 Imbalance Corrected (Data Resampling)", 
        output_filename="cm_resampling.png"
    )