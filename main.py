from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cv2
import numpy as np
import io
from PIL import Image
import os
import uvicorn
import joblib
import tensorflow as tf
from typing import Optional
import json

app = FastAPI(
    title="Facial Recognition API",
    description="Live facial emotion recognition and analysis",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (optional)
security = HTTPBearer()
API_KEYS = {
    "demo-key-123": "demo-user",  # Replace with your actual API keys
    "frontend-key-456": "frontend-user"
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Load models at startup
@app.on_event("startup")
async def load_models():
    global model, label_encoder, scaler
    try:
        # Load your trained models
        model = joblib.load("app/model_files/model_fer.pkl")
        label_encoder = joblib.load("app/model_files/label_encoder_fer.pkl")
        scaler = joblib.load("app/model_files/scaler_fer.pkl")
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        model = None
        label_encoder = None
        scaler = None

@app.get("/")
def root():
    return {
        "message": "Facial Recognition API is running!",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": model is not None,
        "message": "API is running properly"
    }

@app.post("/predict")
async def predict_emotion(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Your facial recognition logic here
        # This is where you'd integrate your existing model prediction code
        
        # Example prediction (replace with your actual model logic)
        if model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Placeholder - integrate your actual prediction logic
        prediction = predict_facial_emotion(opencv_image)
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": 0.95,  # Your actual confidence score
            "message": "Emotion detected successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def predict_facial_emotion(image):
    """
    Integrate your existing facial recognition logic here
    This should return the predicted emotion
    """
    # Your existing prediction code from webcam_inference.py
    # or final_facial_model.py goes here
    
    # Placeholder return
    return {
        "emotion": "happy",
        "features": "detected",
        "face_detected": True
    }

@app.post("/predict-no-auth")
async def predict_no_auth(file: UploadFile = File(...)):
    """Endpoint without API key for testing"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        prediction = predict_facial_emotion(opencv_image)
        
        return {
            "status": "success",
            "prediction": prediction,
            "message": "Emotion detected successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# For live video stream (WebSocket endpoint)
@app.websocket("/ws/live-feed")
async def websocket_endpoint(websocket):
    await websocket.accept()
    try:
        while True:
            # Handle live video stream
            data = await websocket.receive_bytes()
            # Process frame and send prediction back
            # Your live processing logic here
            await websocket.send_json({"prediction": "live_emotion"})
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)