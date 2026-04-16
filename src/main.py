import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Literal

from prediction import predict_class, load_classes
from train_classifier import load_training_config, load_model

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

CONFIG = load_training_config(
    BASE_DIR / 'config/training.yaml',
    BASE_DIR / 'config/model.yaml'
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = load_model(config=CONFIG,
                    model_name="resnet50",
                    device=DEVICE)

model2 = load_model(config=CONFIG,
                    model_name="efficientnet_b3",
                    device=DEVICE)

MODELS = {"resnet50": model1, "efficientnet_b3": model2}

MAX_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 5))* 1024 * 1024
THRESHOLD = float(os.getenv("THRESHOLD", 0.7))
CLASS_NAMES = load_classes()

app = FastAPI(title="Driver Distraction classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(model_name: Literal["resnet50", "efficientnet_b3"],file: UploadFile = File(...)):

    allowed_extensions = (".jpg", ".jpeg", ".png")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    temp_path = None
    try:
        content = await file.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        file_path = file.filename 
        temp_path = BASE_DIR / "temp" / file_path

        # Creating temp directory
        os.makedirs(BASE_DIR / "temp", exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(content)
        # Prediction
        input_size = CONFIG["models"][model_name]["input_size"]
        results = predict_class(image_path=temp_path,
                                model=MODELS[model_name], 
                                class_names=CLASS_NAMES,
                                input_size=input_size, threshold=THRESHOLD) 
        return JSONResponse(content=results, status_code=200)
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Deleting temp_file
        if  temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))
    reload = os.getenv("RELOAD") == "True"
    uvicorn.run("main:app",host=host, port=port , reload=reload)


