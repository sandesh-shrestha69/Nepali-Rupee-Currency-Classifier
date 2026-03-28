from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import base64
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['Rs 10', 'Rs 100', 'Rs 1000', 'Rs 20', 'Rs 5', 'Rs 50', 'Rs 500']

# Detection threshold
CONFIDENCE_THRESHOLD = 0.70  # Production: 70% confidence required

# Transform
transform = A.Compose([
    A.Resize(224, 224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, fill=(0,0,0)),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════

def load_model():
    print("Loading model...")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 7)
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location=device)
    )
    model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    return model

model = load_model()

# ══════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════

app = FastAPI(
    title="Nepali Currency Classifier",
    description="Real-time currency detection with smart bounding boxes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════

def predict_from_image(image: Image.Image):
    """Predict currency from PIL Image"""
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
    
    # ALWAYS map class name
    currency_name = CLASS_NAMES[predicted_class]
    
    # Determine if currency is detected
    is_detected = confidence >= CONFIDENCE_THRESHOLD
    
    print(f"DEBUG PREDICT: class_id={predicted_class} ({currency_name}), conf={confidence:.3f}, threshold={CONFIDENCE_THRESHOLD}, detected={is_detected}")
    
    return {
        "detected": is_detected,
        "class_id": predicted_class,
        "currency": currency_name,
        "confidence": float(confidence),
        "threshold": CONFIDENCE_THRESHOLD,
        "all_probabilities": {
            CLASS_NAMES[i]: float(prob)
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
    }

# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Nepali Currency Classifier - Smart Detection",
        "version": "2.0.0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload mode prediction"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_from_image(image)
        
        print(f"UPLOAD PREDICT: {result['currency']} ({result['confidence']:.1%}) detected={result['detected']}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """Real-time detection via WebSocket"""
    await websocket.accept()
    print("🔌 WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                image_data = message.get('image')
                
                if not image_data:
                    continue
                
                # Decode base64
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Predict
                result = predict_from_image(image)
                
                # Add bounding box ONLY if currency detected
                if result['detected']:
                    # For now, box covers most of the frame
                    # In a real object detection model, this would be the actual detected region
                    result['bbox'] = {
                        'x': 0.15,      # 15% from left
                        'y': 0.15,      # 15% from top
                        'width': 0.70,  # 70% of frame width
                        'height': 0.70  # 70% of frame height
                    }
                else:
                    result['bbox'] = None  # No box if not detected
                
                await websocket.send_json(result)
                
            except Exception as e:
                print(f"❌ Frame error: {e}")
                await websocket.send_json({
                    "detected": False,
                    "error": str(e),
                    "currency": "Error",
                    "confidence": 0.0
                })
    
    except Exception as e:
        print(f"🔌 WebSocket error: {e}")
    finally:
        print("🔌 WebSocket closed")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)