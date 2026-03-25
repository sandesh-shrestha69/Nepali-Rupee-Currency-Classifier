import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import datasets
from efficientnet_pytorch import EfficientNet  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
from pathlib import Path
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented['image']

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\\n")

    # ENHANCED TRANSFORMS  
    train_aug = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=30, p=0.7),
        A.VerticalFlip(p=0.4),
        A.HorizontalFlip(p=0.5),  
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_data_path = "./data/processed/train"
    val_data_path = "./data/processed/val"

    train_data = datasets.ImageFolder(train_data_path, transform=AlbumentationsWrapper(train_aug))
    val_data = datasets.ImageFolder(val_data_path, transform=AlbumentationsWrapper(val_aug))
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}\\n")

    # ENHANCED CLASS BALANCING - BOOST Rs5 (index 4)
    class_counts = Counter(train_data.targets)
    print("Class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {train_data.classes[class_idx]}: {count} images")
    
    weight_per_class = [1.0 / class_counts[i] for i in range(len(class_counts))]

    print(f"Enhanced weights: {weight_per_class}")
    
    sample_weights = [weight_per_class[label] for label in train_data.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DATA LOADERS
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

    # MODEL
    print("\\nLoading EfficientNet-B0...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(train_data.classes)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model = model.to(device)
    print(f"Model loaded for {num_classes} classes.\\n")

    # FOCAL LOSS for hard examples (Rs5 vs Rs10)
    criterion = FocalLoss(gamma=2.0)
    print("Using FocalLoss (gamma=2.0) for hard example focus.\\n")

    best_val_acc = 0.0

    # PHASE 1: HEAD ONLY (extended)
    print("="*70)
    print("PHASE 1: Train classifier head (5 epochs, lr=1e-3)")
    print("="*70)
    
    optimizer = optim.Adam(model._fc.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/5 - Train: {train_acc:.2f}% ({train_loss/len(train_loader):.4f}), Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('./models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), './models/best_model_fixed.pth')
            print(f"  → Saved best: {best_val_acc:.2f}%")

    # PHASE 2: FULL FINE-TUNE (extended, lower LR)
    print("\\n" + "="*70)
    print("PHASE 2: Fine-tune all layers (25 epochs, lr=3e-5)")
    print("="*70)
    
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    for epoch in range(25):
        # Train loop same as above
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total

        # Val same
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/25 - Train: {train_acc:.2f}% ({train_loss/len(train_loader):.4f}), Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_model_fixed.pth')
            print(f"  → Saved best: {best_val_acc:.2f}%")

    print("\\n✅ FIXED Training complete! Best val acc: {:.2f}%".format(best_val_acc))
    print("New model saved: ./models/best_model_fixed.pth")

if __name__ == "__main__":
    train()

