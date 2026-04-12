# Driver Gesture Detection System

A deep learning pipeline for classifying driver distraction behaviors from dashboard camera images, using transfer learning with ResNet50 and EfficientNet-B3.

---

## Current Vision

### What the project does today

The system trains and evaluates image classifiers to detect **10 driver distraction states**:

| Class | Label |
|-------|-------|
| 0 | Safe driving |
| 1 | Texting (right hand) |
| 2 | Talking on phone (right hand) |
| 3 | Texting (left hand) |
| 4 | Talking on phone (left hand) |
| 5 | Operating radio |
| 6 | Drinking |
| 7 | Reaching behind |
| 8 | Hair / makeup |
| 9 | Talking to passenger |

### Architecture

- **Backbones**: ResNet50 and EfficientNet-B3 (pretrained on ImageNet)
- **Head**: Custom classifier — optional hidden layer (512 units), BatchNorm, ReLU, Dropout, then a 10-class linear output
- **Fine-tuning**: Backbone frozen by default; partial unfreezing supported via config

### Training pipeline

- Optimizer: AdamW (`lr=3e-4`, `weight_decay=5e-4`)
- Loss: CrossEntropyLoss with label smoothing (0.05)
- Scheduler: ReduceLROnPlateau
- Early stopping with configurable patience
- Checkpointing: saves best and last model weights

### Data pipeline (`src/preprocess.py`)

- `DriverDistractionDataset`: loads images from a CSV manifest (`image_path`, `label_id`)
- Augmentations on training: horizontal flip, rotation ±10°, color jitter
- ImageNet normalization for all splits

### Evaluation (`src/evaluate.py`)

- Metrics: Accuracy, Precision, Recall, F1-Score (macro)
- Outputs: confusion matrices, per-class accuracy bars, train/val/test phase comparison
- Saves: PNG visualizations + JSON metrics summary to `results/`

### Current limitations

- **Offline only** — there is no inference script; the trained model cannot be used on new images without manual code
- **No real-time support** — no video/webcam integration
- **No deployment artifact** — model is saved as a raw `.pth` checkpoint, with no export to ONNX / TorchScript
- **No end-to-end pipeline script** — preprocessing, training, and evaluation are orchestrated through notebooks only

---

## Roadmap: Making the System Useful for Prediction

The following enhancements turn this research prototype into a production-ready prediction system, ordered from highest to lowest impact.

### 1. Add a prediction script (`src/predict.py`)

The most critical missing piece. A script that:
- Loads the best checkpoint (`checkpoints/<model>_best.pth`)
- Accepts a single image path or a directory of images
- Applies the same val/test transforms
- Returns the predicted class label and confidence score

```python
# Minimal interface
python src/predict.py --image path/to/frame.jpg --model resnet50
# → Predicted: safe_driving (confidence: 0.94)
```

### 2. Export the model to ONNX / TorchScript

Raw `.pth` files require PyTorch to be installed at inference time. Exporting removes that dependency and enables deployment anywhere:

```python
# TorchScript (no Python overhead)
scripted = torch.jit.script(model)
scripted.save("checkpoints/resnet50.pt")

# ONNX (cross-platform, works with ONNX Runtime, TensorRT, CoreML)
torch.onnx.export(model, dummy_input, "checkpoints/resnet50.onnx", ...)
```

### 3. Real-time webcam / video inference

Wire the prediction logic to OpenCV to process a live camera feed or a recorded video file:

- Read frames with `cv2.VideoCapture`
- Run prediction every N frames (e.g., every 5 frames at 30 fps = 6 predictions/sec)
- Overlay the predicted label and confidence on the frame
- Trigger an alert (sound, on-screen warning) when a distraction class is detected

### 4. Add a `src/run_pipeline.py` end-to-end script

Replace the notebook-only workflow with a single CLI entry point:

```bash
python src/run_pipeline.py --mode train   --model resnet50
python src/run_pipeline.py --mode eval    --model resnet50
python src/run_pipeline.py --mode predict --model resnet50 --input data/test_frames/
```

This makes the project reproducible without Jupyter and enables CI/CD integration.

### 5. Fix the `load_model` key mismatch bug

`save_checkpoint` stores the key `model_state`, but `load_model` reads `model_state_dict`. This will crash at inference time:

```python
# src/train_classifier.py:380 — current (broken)
model.load_state_dict(checkpoint['model_state_dict'])

# Fix: match the key used in save_checkpoint
model.load_state_dict(checkpoint['model_state'])
```

### 6. Confidence thresholding and uncertainty handling

Raw `argmax` predictions can be wrong with high confidence. Add:
- A minimum confidence threshold (e.g., 0.6) — below it, return `"uncertain"`
- Temperature scaling for calibrated probabilities
- Log low-confidence frames for review / retraining

### 7. Lightweight model for edge deployment

EfficientNet-B3 and ResNet50 are large for in-vehicle hardware. Consider:
- **MobileNetV3** or **EfficientNet-B0** — same API, much smaller
- **Post-training quantization** (INT8) via `torch.quantization` — 4× size reduction with minimal accuracy loss
- **Knowledge distillation** — train a small student model from the best checkpoint

### 8. REST API wrapper (FastAPI)

Expose prediction as a microservice so dashcam systems, mobile apps, or fleet management platforms can call it:

```
POST /predict
Content-Type: multipart/form-data
Body: image file

Response: {"class": "texting_right", "confidence": 0.87, "label_id": 1}
```

### 9. Temporal smoothing for video streams

Single-frame predictions are noisy. A sliding window majority vote over the last K frames (e.g., K=5) greatly reduces false positives:

```python
from collections import deque
window = deque(maxlen=5)
window.append(predicted_class)
final_prediction = Counter(window).most_common(1)[0][0]
```

### 10. Data and retraining improvements

- Add the `epoch` key to `save_checkpoint` (currently missing, causing a `KeyError` in `load_model`)
- Support class-weighted loss (already in config, just pass real weights from class frequency analysis)
- Add test-time augmentation (TTA) — average predictions over flipped/rotated versions of each image for better accuracy
