from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


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

# LOAD MODEL AT STARTUP (ONCE!)

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


model = load_model()

app = FastAPI(
    title="Nepali Currency Classifier",
    description="Upload Nepali currency image and get prediction",
    version="1.0.0"
)

# Add CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_from_image(image: Image.Image):
    """Predict currency from PIL Image"""
    
    # Convert to RGB
    image = image.convert('RGB')
    
    # Transform
    img_np = np.array(image)
    augmented = transform(image=img_np)
    image_tensor = augmented['image'].unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "class_id": predicted_class,
        "currency": CLASS_NAMES[predicted_class],
        "confidence": float(confidence),
        "all_probabilities": {
            CLASS_NAMES[i]: float(prob)
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Nepali Currency Classifier API",
        "version": "1.0.0",
        "model": "EfficientNet-B0",
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict currency denomination from uploaded image
    
    Returns:
        - currency: Denomination name (e.g., "Rs 100")
        - confidence: Prediction confidence (0-1)
        - all_probabilities: Dict of all class probabilities
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpeg, png, etc.)"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict
        result = predict_from_image(image)
        
        # Log prediction
        print(f"\n{'='*50}")
        print(f"🎯 Predicted: {result['currency']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"{'='*50}\n")
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)