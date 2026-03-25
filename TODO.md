# Fix Rs10 vs Rs5 Model Confusion

## Current Status
- Data: train/ has numeric dirs ('10': ~107k imgs, '5': ~53k imgs, '100': ~114k, etc.). Imbalance: Rs5 has ~50% fewer than Rs10.
- Classes: Alphabetical ImageFolder → index 0='10' (Rs10), 4='5' (Rs5).
- Model: EfficientNet-B0, likely confusing due to fewer Rs5 samples + visual similarity.

## TODO Steps

### Step 1: Diagnosis ✅
- [x] Data structure & counts (Rs5:1400/8.8%, Rs10:2900/18.3%)
- [x] prepare_data.py run
- [RUNNING] evaluate.py → await CM
- [ ] Test user Rs5 image

### Step 2: Data Fixes
- Custom oversampler for Rs5 (weight=2.0 for class 4)
- Verify splits with splitfolders if needed

### Step 3: Enhanced Training 🔄
- [x] Created scripts/train_fixed.py (FocalLoss, Rs5 boost 1.8x, stronger aug, 5+25 epochs, lr=3e-5)
- [ ] Run `python scripts/train_fixed.py`
- [ ] Verify new model
  | Change | Why |
  |--------|-----|
  | FocalLoss | Handle hard examples (Rs5/10) |
  | Custom sampler: Rs5 weight*2 | Balance low-sample class |
  | Aug: Add ShiftScaleRotate(p=0.5), CoarseDropout | Better generalization |
  | Phase1: 5 epochs, Phase2: 25 epochs, lr=3e-5 | More training |
  | Save './models/best_model_fixed.pth'
- Run `python scripts/train.py`

### Step 4: Update Dependent Files
- scripts/predict.py: Load fixed_model, temperature scaling (T=1.5 for calibration)
- scripts/main.py, FastAPI: Update model path
- scripts/evaluate.py: Load fixed model

### Step 5: Validation
- Re-run evaluate.py → check improved F1 for '5'
- Retest user Rs5 image → expect class_id=4 high conf

### Step 6: Cleanup
- Update TODO [live]
- attempt_completion if resolved

**Progress: Step 1 → Step 3 after eval complete.**
