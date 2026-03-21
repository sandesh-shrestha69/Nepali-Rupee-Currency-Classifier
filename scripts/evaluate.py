import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from efficientnet_pytorch import EfficientNet
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


# WRAPPER FOR ALBUMENTATIONS
class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented['image']


# SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# TEST TRANSFORM (MUST MATCH TRAINING!)
test_transform = A.Compose([
    A.Resize(224, 224),  
    A.Normalize(         
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# LOAD MODEL
def load_model(num_classes):
    print("Loading model...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load('./models/best_model.pth'))
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    return model


# LOAD TEST DATA
test_data = "/processed/test"
if os.path.exists(test_data):
    print(f" Data path is ready: {test_data}")
    print("Files found:", os.listdir(test_data) [:5]) # Shows first 5 files
else:
    print(f" Error: {test_data} not found. Check your unzip step.")

test_data = datasets.ImageFolder(
    test_data, 
    transform=AlbumentationsWrapper(test_transform)  
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print(f"Test samples: {len(test_data)}")
print(f"Classes: {test_data.classes}\n")

num_classes = len(test_data.classes)


# EVALUATE


model = load_model(num_classes)

print("Evaluating...\n")

all_pred = []
all_labels = []
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_pred.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# RESULTS


test_acc = 100 * test_correct / test_total

print("="*60)
print(f"📊 Test Accuracy: {test_acc:.2f}%")
print("="*60)


# CONFUSION MATRIX


cm = confusion_matrix(all_labels, all_pred)

print("\n🔢 Confusion Matrix:")
print("           Predicted")

# Create header with class names
header = "           " + "  ".join([f"{cls:>4}" for cls in test_data.classes])
print(header)

# Print matrix with actual class names
for i, class_name in enumerate(test_data.classes):
    row = f"Actual {class_name:>4}: {cm[i]}"
    print(row)


# CLASSIFICATION REPORT


print("\n Classification Report:")
print(classification_report(
    all_labels,
    all_pred,
    target_names=test_data.classes,  
    digits=4
))

print("="*60)
print(" Evaluation completed!")
print("="*60)