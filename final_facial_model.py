import cv2
import joblib
import time
import json
from microexpression_tracker import track_microexpressions, get_lip_engagement
from collections import Counter
import mediapipe as mp
import numpy as np

# --- Load microexpression calibration ---
with open('user_calibration.json', 'r') as f:
    calibration_ref = json.load(f)

# --- Load your ELM model and scaler ---
# (Commented out for now)
# elm_model = joblib.load("src/model_fer.pkl")
# scaler = joblib.load("src/scaler_fer.pkl")

IMG_SIZE = 96  # use same as in training
SESSION_DURATION = 15  # seconds
EYE_AWAY_THRESHOLD = 20
HEAD_TURN_THRESHOLD = 20

def preprocess_for_model(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.flatten().reshape(1, -1)
    return img

cap = cv2.VideoCapture(0)

print("\n📸 Welcome! Press SPACE to start stress & engagement analysis.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.putText(frame, "Press SPACE to start", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break

print("\n🟡 Session started. Look at the screen for 15 seconds.")
session_start = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

eye_away_count = 0
head_turn_count = 0
engagement_predictions = []  # Not used if ELM is commented out, but kept for completeness
lip_engagement_predictions = []

while time.time() - session_start < SESSION_DURATION:
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Microexpression detection ---
    micro, face_bbox, multiple_faces = track_microexpressions(frame, face_mesh, calibration_ref)

    # --- Lip engagement detection ---
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    lip_engagement = "No Face"
    if results.multi_face_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        lip_engagement = get_lip_engagement(landmarks)
        lip_engagement_predictions.append(lip_engagement)
        if face_bbox:
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0,255,0), 2)
    else:
        lip_engagement_predictions.append("No Face")
        landmarks = None

    if micro["eye_away"]:
        eye_away_count += 1
    if micro["head_turn"]:
        head_turn_count += 1

    if multiple_faces:
        cv2.putText(frame, "Multiple faces detected!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    elapsed = int(time.time() - session_start)
    cv2.putText(frame, f"Time left: {SESSION_DURATION-elapsed}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Engagement: {lip_engagement}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('SafeSpace Session', frame)

    print(f"Frame: {len(lip_engagement_predictions)} | Engagement: {lip_engagement}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Summarize Lip Engagement Results (Safe & Robust) ---
lip_counts = Counter(lip_engagement_predictions)
# Remove "No Face" for reporting majority, if present
lip_counts_no_face = Counter({k: v for k, v in lip_counts.items() if k != "No Face"})
lip_total = sum(lip_counts_no_face.values())

if lip_total > 0:
    for label in ["Engaged", "Partially Engaged", "Not Engaged"]:
        print(f"Engagement {label}: {lip_counts_no_face.get(label,0)} frames ({(lip_counts_no_face.get(label,0)/lip_total)*100:.1f}%)")

    lip_majority_label = lip_counts_no_face.most_common(1)[0][0]
else:
    print("No valid engagement predictions to summarize.")
    lip_majority_label = "No Face"

# --- Microexpression-based feedback ---
if 10 < eye_away_count < 20 and 10 < head_turn_count < 20:
    print(f"\nEye distraction detected some times in session.")
    print(f"Head turn detected some times in session.")

if eye_away_count > EYE_AWAY_THRESHOLD:
    print(f"\nEye distraction detected many times in session.")
if head_turn_count > HEAD_TURN_THRESHOLD:
    print(f"Head turn detected many times in session.")

# --- Final Hybrid Result (using only lips & microexpression, as ELM is off) ---
hybrid_result = lip_majority_label

# Apply microexpression override
if eye_away_count > EYE_AWAY_THRESHOLD or head_turn_count > HEAD_TURN_THRESHOLD:
    if hybrid_result == "Engaged":
        hybrid_result = "Partially Engaged"
    elif hybrid_result == "Partially Engaged":
        hybrid_result = "Not Engaged"

print(f"\n✅ Final Conclusion (with Microexpressions): {hybrid_result}")
