import cv2
import time
import mediapipe as mp
import numpy as np
import json
from microexpression_tracker import track_microexpressions

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)
CALIBRATION_DURATION = 10 # seconds

print("\n🟩 Welcome to SafeSpace!")
print("Press SPACE to begin calibration. Position your face within the webcam frame and keep it still.")
cap = cv2.VideoCapture(0)

# Wait for SPACE key to start
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.putText(frame, "Press SPACE to start calibration", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break

print("\n🟡 Calibration started. Please follow the instructions for 5 seconds.")

start_time = time.time()
eye_centers = []

while time.time() - start_time < CALIBRATION_DURATION:
    ret, frame = cap.read()
    if not ret:
        continue

    micro, face_bbox, multiple_faces = track_microexpressions(frame, face_mesh, {})

    h, w, _ = frame.shape

    if face_bbox:
        cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0,255,0), 2)
    if multiple_faces:
        cv2.putText(frame, "Multiple faces detected!", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if face_bbox:
        eye_centers.append((face_bbox[0] + face_bbox[2]) / 2 / w)

    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Calibration: {CALIBRATION_DURATION-elapsed}s left", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n✅ Calibration finished. Thank you!")
cap.release()
cv2.destroyAllWindows()

user_calib = {
    'eye_left': np.percentile(eye_centers, 5) if eye_centers else 0.35,
    'eye_right': np.percentile(eye_centers, 95) if eye_centers else 0.65
}

with open('user_calibration.json', 'w') as f:
    json.dump(user_calib, f)

print("\n🟢 Ready for session. Please start the main stress detection.")
