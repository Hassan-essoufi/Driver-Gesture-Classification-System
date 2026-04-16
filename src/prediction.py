import yaml
from pathlib import Path
import torch
import torch.nn.functional as F

from preprocess import load_image, get_val_transforms

def load_classes():
    """
    Loding class mapping
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    mapping_path = BASE_DIR / 'config/class_mapping.yaml'

    try:
        with open(mapping_path,'r') as mapping:
            config = yaml.safe_load(mapping)
    
        return config['classes']
    except Exception as e:
        raise Exception(f"error in loading configuration: {e}")
 
def preprocess_image(image_path, input_size=(224,224)):
    """
      preprocessing an image and trosform it to tensor ready for model
    """
    try:
        image = load_image(image_path)
        transform = get_val_transforms(input_size)
        new = transform(image).unsqueeze(0)
        return new
        
    except Exception as e:
        raise Exception(f" error when loading image: {e}")

def run_inference(model, tensor, device):
    """
    predict the class of an image (inference).
    """
    try:
        if model:
            model.to(device)
            model.eval()

        with torch.no_grad():
            tensor = tensor.to(device)

            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            scores, preds = torch.max(probs, dim=1)

            best_idx = preds[0]
            best_score = scores[0]

            return best_idx, best_score
    except Exception as e:
        raise Exception(f"error in inference: {e}")
    
def interpret_prediction(class_idx, confidence, class_names, threshold):
    """
    final class decision based on threshold.
    """
    try:
        idx = class_idx.item() if hasattr(class_idx, 'item') else int(class_idx)
        conf = confidence.item() if hasattr(confidence, 'item') else float(confidence)

        if conf <= threshold:
            return {"class": "uncertain", "label_id": idx, "confidence": conf}

        current_class = class_names[idx]
        return {"class": current_class, "label_id": idx, "confidence": conf}

    except Exception as e:
        raise Exception(f"Something went wrong: {e}")

def predict_class(image_path, model, class_names, input_size, threshold):
    """
    prediction function of an image's class
    """
    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = preprocess_image(image_path, input_size)
        
        best_idx, best_score = run_inference(model, tensor, device)
        interpretation = interpret_prediction(best_idx, best_score, class_names, threshold)
        
        return interpretation
    
    except Exception as e:
        raise Exception(f"Error while predicting: {e}")












