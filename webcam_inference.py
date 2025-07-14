# webcam_inference.py

import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from utils import map_emotion_to_engagement
import os
from hpelm import ELM
import joblib
from simple_elm import SimpleELMClassifier
import joblib
model_fer = joblib.load("src/model_fer.pkl")
scaler_fer = joblib.load("src/scaler_fer.pkl")

# Load models and scalers
# elm=ELM(2048,7)
# model_fer = joblib.load("src/model_fer.pkl")
# scaler_fer = joblib.load("src/scaler_fer.pkl")
# model_ck = joblib.load("src/model_ck.pkl")
# scaler_ck = joblib.load("src/scaler_ck.pkl")

# Class index to emotion label
class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# Load ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3), pooling='avg')

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (96, 96))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return preprocess_input(frame)

# Predict function with neutral class bypass logic
def predict_emotion(frame):
    try:
        # Extract features
        preprocessed = preprocess_frame(frame)
        features = resnet.predict(preprocessed, verbose=0).flatten().reshape(1, -1)

        # Scale features
        feat_fer = scaler_fer.transform(features)
        feat_ck = scaler_ck.transform(features)

        # Get probabilities
        probs_fer = model_fer.predict_proba(feat_fer)[0]
        # probs_ck = model_ck.predict_proba(feat_ck)[0]

        label_fer = class_names[np.argmax(probs_fer)]

        # If FER predicts neutral, use only FER
        final_probs = probs_fer

        final_label = class_names[np.argmax(final_probs)]
        return final_label

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "error"

# Main webcam loop
def run_webcam():
    cap = cv2.VideoCapture(0)
    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)  # Flip for mirror view
        emotion = predict_emotion(flipped)
        engagement = map_emotion_to_engagement(emotion)

        # Overlay
        cv2.putText(flipped, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(flipped, f"Engagement: {engagement}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FINAL FACIAL MODEL FOR SAFESPACE", flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
