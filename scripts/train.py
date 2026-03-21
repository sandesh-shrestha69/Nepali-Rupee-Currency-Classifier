import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import datasets
from efficientnet_pytorch import EfficientNet  
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
from pathlib import Path

class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented['image']

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # TRANSFORMS  
    train_aug = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.3),  
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.8
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    val_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

    # LOAD DATA
    train_data = datasets.ImageFolder(
        "./data/processed/train", 
        transform=AlbumentationsWrapper(train_aug)
    )
    val_data = datasets.ImageFolder(
        './data/processed/val', 
        transform=AlbumentationsWrapper(val_aug)
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}\n")

    # CLASS BALANCING
    class_counts = Counter(train_data.targets)
    print("Class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {train_data.classes[class_idx]}: {count} images")
    
    weight_per_class = [1.0/class_counts[i] for i in range(len(class_counts))]
    sample_weight = [weight_per_class[label] for label in train_data.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(sample_weight),
        replacement=True
    )
    
    # DATA LOADERS
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=2
    )
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # MODEL SETUP    
    print("\nLoading EfficientNet-B0...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_classes = len(train_data.classes)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    
    model = model.to(device)
    print(f"Model loaded. Training {num_classes} classes.\n")

    # PHASE 1: TRAIN HEAD ONLY 
    print("="*60)
    print("PHASE 1: Training head only (3 epochs)")
    print("="*60)
    
    optimizer = optim.Adam(model._fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(3):
        # ─── Training ───
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()  
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ─── Validation ───
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/3")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:  
            best_val_acc = val_acc
            Path('./models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), './models/best_model.pth')
            print(f"  ✅ Saved (best: {best_val_acc:.2f}%)")

    # PHASE 2: FINE-TUNE ALL LAYERS    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning all layers (15 epochs)")
    print("="*60)
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    # Small learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(15):
        # ─── Training ───
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ─── Validation ───
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/15")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:  
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_model.pth')
            print(f" Saved (best: {best_val_acc:.2f}%)")

    print("\n" + "="*60)
    print(f"Training complete! Best accuracy: {best_val_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    train()