import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
import joblib
from simple_elm import SimpleELMClassifier
import cv2
from microexpression_tracker import track_microexpressions
import time
import json

class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3), pooling='avg')

model_fer = joblib.load("src/model_fer.pkl")
scaler_fer = joblib.load("src/scaler_fer.pkl")
# model_ck = joblib.load("src/model_ck.pkl")
# scaler_ck = joblib.load("src/scaler_ck.pkl")

def map_emotion_to_engagement(emotion_label):
    # fer returns: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    if emotion_label in ['happy', 'surprise']:
        return "Engaged"
    elif emotion_label in ['neutral']:
        return "Partially Engaged"
    else:
        return "Not Engaged"


def preprocess_image_for_resnet(image, target_size=(96, 96)):
    import cv2
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def predict_emotion(frame):
    try:
        preprocessed = preprocess_image_for_resnet(frame)
        features = resnet.predict(preprocessed, verbose=0).flatten().reshape(1, -1)
        feat_fer = scaler_fer.transform(features)
        probs_fer = model_fer.predict_proba(feat_fer)[0]
        # Get the index of the class with highest probability
        idx = np.argmax(probs_fer)
        final_label = class_names[idx]
        print(f"[DEBUG] Emotion probabilities: {dict(zip(class_names, np.round(probs_fer, 3)))}")
        print(f"[DEBUG] Detected emotion: {final_label}")
        return final_label
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "error"


def predict_engagement_class(frame):
    emotion_label = predict_emotion(frame)
    print(f"[DEBUG] Detected emotion: {emotion_label}")
    engagement_label = map_emotion_to_engagement(emotion_label)
    return engagement_label


