from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['Rs 10', 'Rs 100', 'Rs 1000', 'Rs 20', 'Rs 5', 'Rs 50', 'Rs 500']

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(         
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

def load_model():
    print("Loading model...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 7)
    model.load_state_dict(
        torch.load('./models/best_model.pth', map_location=device)
    )
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    return model

# Load model once
model = load_model()

def predict_currency(image_path):
    """Predict currency from image path"""
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    print("Transforming...")
    img_np = np.array(image)
    image_tensor = transform(image=img_np)['image'].unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    print("Predicting...")
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "currency": predicted_class,
        "confidence": confidence,
        "all_probabilities": probabilities[0].cpu().numpy()
    }

if __name__ == '__main__':
    # Test with an image
    result = predict_currency("WhatsApp Image 2026-03-20 at 10.03.21 AM.jpeg")
    
    print(f"\n{'='*60}")
    print(f"🎯 Predicted: {CLASS_NAMES[result['currency']]}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"{'='*60}")
    
    print(f"\n📈 All Probabilities:")
    for i, prob in enumerate(result['all_probabilities']):
        bar = '█' * int(prob * 50)
        print(f"  {CLASS_NAMES[i]:>8}: {prob:.4f} {bar}")
    print()