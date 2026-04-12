import torch
import torch.nn.functional as F
import yaml
from preprocess import load_image, get_val_transforms
from train_classifier import load_training_config


def load_classes(mapping_path='config/class_mapping.yaml'):
    """
    Loding class mapping
    """
    try:
        with open(mapping_path,'r') as mapping:
            config = yaml.safe_load(mapping)
    
        return {** config}
    except Exception as e:
        raise {f"error in loading configuration: {e}"}
 
def preprocess_image(image_path, input_size=(124,124)):
    """
      preprocessing an image and trosform it to tensor ready for model
    """
    try:
        image = load_image(image_path)
        if image is not None:
            transform = get_val_transforms(input_size)
            return transform(image)
        
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

            best_idx = torch.argmax(scores)
            best_class = preds[best_idx]
            best_score = scores[best_idx]

            return best_class, best_score
    except Exception as e:
        raise Exception(f"error in inference: {e}")
    
def interpret_prediction(class_idx, confidence, class_names, threshold):
    """
    final class decision based on threshold.
    """    
    try:
        current_class =class_names[class_idx]
        if confidence <= threshold:
            current_class = "uncertain"
        
        return current_class
    except Exception as e:
        raise Exception(f"Something went error: {e}")

    





