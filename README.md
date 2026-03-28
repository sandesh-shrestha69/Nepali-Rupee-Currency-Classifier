# 🇳🇵 Nepali Currency Classifier - Real-Time Detection System

A production-grade real-time currency recognition system achieving **99% accuracy** on 3,401 test images using EfficientNet transfer learning and computer vision techniques.

![Demo](https://img.shields.io/badge/Accuracy-99%25-success)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)

---

## 🌟 Features

### Core Capabilities
- ✅ **Real-time Detection**: Continuous scanning via WebSocket (500ms intervals)
- ✅ **Smart Bounding Boxes**: Only displays when currency detected (>70% confidence)
- ✅ **High Accuracy**: 99% test accuracy on 3,401 images
- ✅ **Multi-denomination Support**: 7 Nepali currency denominations (Rs 5, 10, 20, 50, 100, 500, 1000)
- ✅ **Dual Mode**: Upload image or live camera detection
- ✅ **Production Ready**: FastAPI backend with WebSocket streaming

### Technical Highlights
- **Transfer Learning**: EfficientNet-B0 pre-trained on ImageNet
- **Data Augmentation**: Albumentations library with rotation, brightness, contrast, blur
- **Class Balancing**: WeightedRandomSampler for imbalanced datasets
- **Two-Phase Training**: Head-only (3 epochs) → Full fine-tuning (15 epochs)
- **Real-time Inference**: WebSocket-based continuous detection
- **Confidence Thresholding**: Smart detection with 70% threshold

---

## 📊 Performance Metrics

### Test Results
```
Total Test Samples: 3,401
Overall Accuracy: 99.00%
Total Confusions: 32 (0.94% error rate)

Per-Class Performance:
├─ Rs 5:    300/300 = 100.00% ✅ PERFECT
├─ Rs 100:  554/554 = 100.00% ✅ PERFECT
├─ Rs 10:   599/623 = 96.15% (24 confused with Rs 20)
├─ Rs 1000: 477/479 = 99.58%
├─ Rs 20:   455/463 = 98.27%
├─ Rs 50:   [high accuracy]
└─ Rs 500:  [high accuracy]

Main Confusion Pattern:
  Rs 10 ↔ Rs 20: 32 confusions (expected - similar appearance)
```

### Training Configuration
```
Architecture: EfficientNet-B0
Input Size: 224×224×3
Parameters: ~5M (EfficientNet) + Custom head (7 classes)
Optimizer: Adam (lr=1e-3 for head, 1e-4 for full)
Loss: CrossEntropyLoss
Batch Size: 32
Training Time: ~18 epochs total
Device: CUDA (GPU) / CPU compatible
```

---

## 🏗️ Architecture

### Model Pipeline
```
Input Image (any size)
    ↓
Resize to 224×224
    ↓
Normalize (ImageNet stats)
    ↓
EfficientNet-B0 Backbone (frozen → unfrozen)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (1280 → 7)
    ↓
Softmax Activation
    ↓
Class Probabilities [7]
```

### Training Strategy
```
Phase 1 (5 epochs):
  - Freeze: All EfficientNet layers
  - Train: Only FC layer
  - LR: 1e-3
  - Purpose: Learn task-specific features

Phase 2 (25 epochs):
  - Unfreeze: All layers
  - Train: Full model
  - LR: 1e-4
  - Purpose: Fine-tune entire network
```

---


## 📁 Project Structure
```
Nepali-Rupee-Currency-Classifier/
├── data/
│   |
│   └── processed/              # Split data
│       ├── train/              # 70% (with augmentation)
│       ├── val/                # 15% (no augmentation)
│       └── test/               # 15% (no augmentation)
│
├── models/
│   └── best_model.pth          # Trained weights (99% accuracy)
│
├── scripts/
|   |── __init__.py
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Testing & metrics
│   └── predict.py              # Single image inference
│
├── frontend/
│   ├── index.html              # Web interface
│   ├── style.css               # Styling
│   └── script.js               # Real-time detection logic
│
├── main.py                     # FastAPI server + WebSocket
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
Webcam (for real-time detection)
```

### Installation

1. **Clone repository**
```bash
git https://github.com/sandesh-shrestha69/Nepali-Rupee-Currency-Classifier.git
cd Nepali-Rupee-Currency-Classifier

```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Download Dataset**
make sure you download the dataset in data folder
https://www.kaggle.com/datasets/nawich/nepali-currency

4. **Download model** (if not training from scratch)
```bash
# Model file: models/best_model.pth
# Size: ~20MB
```

---

## 💻 Usage

### Training from Scratch


1. **Train model**
```bash
python scripts/train.py
```

Output:
```
Phase 1: Training head only (3 epochs)
Epoch 1/3: Val Acc: 85.23%
Epoch 2/3: Val Acc: 91.45%
Epoch 3/3: Val Acc: 94.12%

Phase 2: Fine-tuning all layers (15 epochs)
Epoch 1/15: Val Acc: 95.67%
...
Epoch 15/15: Val Acc: 99.12%

✅ Best model saved: 99.12%
```

### Evaluation
```bash
python scripts/evaluate.py
```

Output: Confusion matrix, classification report, per-class metrics

### Single Image Prediction
```bash
python scripts/predict.py
```

### Web Interface

1. **Start backend**
```bash
python main.py
# Server running on http://localhost:8000
```

2. **Start frontend**
```bash
cd frontend
python -m http.server 8080
# Open http://localhost:8080
```

3. **Use the app**
   - **Upload Mode**: Click/drag image → Analyze
   - **Live Detection**: Start camera → Real-time scanning

---

## 🔧 API Documentation

### REST Endpoints

#### `GET /`
Health check
```json
{
  "status": "online",
  "version": "2.0.0",
  "classes": ["Rs 10", "Rs 100", ...]
}
```

#### `POST /predict`
Upload image for prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "detected": true,
  "currency": "Rs 100",
  "confidence": 0.9987,
  "all_probabilities": {
    "Rs 10": 0.0001,
    "Rs 100": 0.9987,
    ...
  }
}
```

### WebSocket Endpoint

#### `WS /ws/detect`
Real-time detection stream

**Send:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Receive:**
```json
{
  "detected": true,
  "currency": "Rs 100",
  "confidence": 0.9543,
  "bbox": {
    "x": 0.15,
    "y": 0.15,
    "width": 0.70,
    "height": 0.70
  }
}
```

---

## 🎨 Data Collection Strategy

### Photo Collection Guidelines

**Per denomination: 50+ photos**

**Variations to capture:**
- ✅ Different angles (0°, 15°, 30°, 45°)
- ✅ Different lighting (bright, dim, outdoor, indoor)
- ✅ Different backgrounds (white, dark, wood, fabric)
- ✅ Different states (flat, crumpled, old, new)
- ✅ Both sides of notes
- ✅ Partial views (for robustness)

**Augmentation applied during training:**
- Rotation: ±30°
- Brightness/Contrast: ±20%
- Gaussian Blur: σ=3
- Horizontal Flip: 30% probability
- Normalization: ImageNet statistics

---

## 🧪 Technical Details

### Data Augmentation
```python
Training Transform:
  - Resize(224, 224)
  - Rotate(limit=30, p=0.7)
  - HorizontalFlip(p=0.3)
  - RandomBrightnessContrast(0.2, 0.2, p=0.8)
  - GaussianBlur(blur_limit=3, p=0.2)
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Validation/Test Transform:
  - Resize(224, 224)
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Class Balancing
```python
Method: WeightedRandomSampler
  - Calculate inverse frequency weights
  - Oversample minority classes
  - Ensures balanced batches
  - No data duplication on disk
```

### Confidence Thresholding
```python
Detection Logic:
  if confidence >= 0.70:
    - Show bounding box
    - Display currency name
    - Green box (high confidence)
  else:
    - No box shown
    - Status: "Scanning..."
    - Wait for better detection
```

---

## 🌐 Deployment

### Render (Free Tier)

1. Push to GitHub
2. Create Render web service
3. Connect repository
4. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy!

## 📦 Dependencies
```
Core ML:
  - torch==2.1.0
  - torchvision==0.16.0
  - efficientnet-pytorch==0.7.1
  - albumentations==1.3.1

Backend:
  - fastapi==0.104.1
  - uvicorn[standard]==0.24.0
  - python-multipart==0.0.6
  - websockets==12.0

Utilities:
  - pillow==10.1.0
  - numpy==1.24.3
  - scikit-learn==1.3.2
```

---

## 🎯 Use Cases

1. **Accessibility Tool**: Help visually impaired identify currency
2. **Point-of-Sale Systems**: Automated currency verification
3. **Education**: Teaching currency recognition
4. **Mobile Apps**: Currency scanner application
5. **Retail**: Quick denomination checking

---

## 🐛 Known Limitations

1. **Bounding Box**: Current box is frame-centered (not true object detection)
   - **Solution**: Upgrade to YOLO for precise localization
   
2. **Single Currency**: Only detects one note at a time
   - **Solution**: Multi-object detection with YOLO
   
3. **Lighting Sensitivity**: Very dim lighting may reduce accuracy
   - **Solution**: More low-light training data
   
4. **Partial Views**: Edge cases with <30% of note visible
   - **Solution**: Additional partial-view augmentation

---

## 🔮 Future Enhancements

- [ ] YOLO integration for true object detection
- [ ] Multi-currency detection (multiple notes in frame)
- [ ] Counterfeit detection features
- [ ] Mobile app (React Native / Flutter)
- [ ] Offline mode (TensorFlow Lite)
- [ ] Voice feedback for accessibility
- [ ] Historical tracking (total counted)
- [ ] Support for coins
- [ ] Multi-language interface

---

## 📚 Learning Resources

### Transfer Learning
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)

### Data Augmentation
- [Albumentations Docs](https://albumentations.ai/docs/)
- [Augmentation Strategies](https://www.kaggle.com/code/grfiv4/plot-confusion-matrix)

### FastAPI & WebSockets
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Tutorial](https://fastapi.tiangolo.com/advanced/websockets/)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional currency denominations
- Counterfeit detection algorithms
- Mobile deployment guides
- Performance optimizations
- Better bounding box algorithms

---

## 📄 License

MIT License - Feel free to use for educational or commercial purposes

---

## 👤 Author

**Your Name**
- GitHub: https://github.com/sandesh-shrestha69/Nepali-Rupee-Currency-Classifier.git
---

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- Albumentations library by Buslaev et al.
- FastAPI framework by Sebastián Ramírez
- PyTorch team for excellent ML framework

---

## 📊 Project Stats
```
Lines of Code: ~1,500
Training Time: 2-3 hours (GPU)
Model Size: 20MB
Inference Speed: 
  - GPU: ~20ms per image
  - CPU: ~200ms per image
Dataset Size: 3,401+ images
Accuracy: 99.00%
False Positive Rate: 0.94%
```

---

**Built with ❤️ for accessibility and education**

*Making currency recognition accessible to everyone*
```

---
