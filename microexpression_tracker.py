import numpy as np
import cv2
import mediapipe as mp


LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
NOSE = 1

mp_face_mesh = mp.solutions.face_mesh
def get_lip_engagement(landmarks):
    TOP_LIP = 13
    BOTTOM_LIP = 14
    LIP_LEFT = 78
    LIP_RIGHT = 308
    top_lip = landmarks[TOP_LIP]
    bottom_lip = landmarks[BOTTOM_LIP]
    left_corner = landmarks[LIP_LEFT]
    right_corner = landmarks[LIP_RIGHT]
    lip_opening = abs(top_lip[1] - bottom_lip[1])
    lip_width = abs(right_corner[0] - left_corner[0])

    # print(f"[DEBUG] lip_opening: {lip_opening:.3f}, lip_width: {lip_width:.3f}")

    # Example, adjust as per your actual values!
    # This logic: high opening OR high width = Engaged (smile/mouth open)
    # very small both = Not Engaged, everything else = Partially Engaged
    if lip_opening > 0.01 or lip_width > 0.18:
        return "Engaged"
    elif lip_opening < 0.002 or lip_width < 0.04:
        return "Not Engaged"
    else:
        return "Partially Engaged"





def track_microexpressions(frame, face_mesh, calibration_ref=None):
    if calibration_ref is None:
        calibration_ref = {}
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    micro = {
             "eye_away": False,
             "head_turn": False,
             }
    face_bbox = None
    multiple_faces = False

    if results.multi_face_landmarks:
        if len(results.multi_face_landmarks) > 1:
            multiple_faces = True

        lm = results.multi_face_landmarks[0].landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        xmin, xmax = min(xs)*w, max(xs)*w
        ymin, ymax = min(ys)*h, max(ys)*h
        face_bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

        eye_x = (lm[LEFT_EYE[0]].x + lm[RIGHT_EYE[0]].x) / 2
        nose_x = lm[NOSE].x

        margin = 0.07
        eye_left_th = calibration_ref.get('eye_left', 0.30)
        eye_right_th = calibration_ref.get('eye_right', 0.70)
        if eye_x < (eye_left_th - margin) or eye_x > (eye_right_th + margin):
            micro["eye_away"] = True
        if nose_x < (eye_left_th - margin) or nose_x > (eye_right_th + margin):
            micro["head_turn"] = True

    return micro, face_bbox, multiple_faces
