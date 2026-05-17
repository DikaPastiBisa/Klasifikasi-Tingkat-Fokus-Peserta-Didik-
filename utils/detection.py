import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# =========================================
# LOAD HAAR CASCADE (backup)
# =========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================================
# LOAD YOLO FACE
# =========================================
yolo_face = YOLO("yolov8n.pt")

# =========================================
# LOAD MODEL
# =========================================
def load_mobilenet(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def load_yolo(path):
    return YOLO(path)

# =========================================
# PREPROCESS MOBILENET
# =========================================
def preprocess(face):
    face = cv2.resize(face, (224, 224))
    face = preprocess_input(face)
    face = np.expand_dims(face.astype(np.float32), axis=0)
    return face

# =========================================
# PREDICT MOBILENET
# =========================================
def predict_mobilenet(interpreter, face):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    conf = float(np.max(output))
    label = "Fokus" if np.argmax(output) == 0 else "Tidak Fokus"

    return label, conf

# =========================================
# 🔥 PREDICT YOLO (FIX TOTAL)
# =========================================
def predict_yolo(model, face):
    try:
        # resize biar stabil
        face = cv2.resize(face, (224, 224))

        results = model(face)
        r = results[0]

        # ======================
        # CASE 1: CLASSIFICATION MODEL
        # ======================
        if hasattr(r, "probs") and r.probs is not None:
            probs = r.probs.data.cpu().numpy()

            cls = int(np.argmax(probs))
            conf = float(np.max(probs))

            label = "Fokus" if cls == 0 else "Tidak Fokus"
            return label, conf

        # ======================
        # CASE 2: DETECTION MODEL
        # ======================
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            conf = float(r.boxes.conf[0])
            cls = int(r.boxes.cls[0])

            label = "Fokus" if cls == 0 else "Tidak Fokus"
            return label, conf

        return "Tidak Fokus", 0.0

    except:
        return "Tidak Fokus", 0.0


# =========================================
# DETECT FACE (YOLO + HAAR)
# =========================================
def detect_face(frame):

    # ================= YOLO =================
    results = yolo_face(frame)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        padding = 20
        h, w = frame.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2, y2)

    # ================= HAAR (backup) =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    padding = 60
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_frame, x + w + padding)
    y2 = min(h_frame, y + h + padding)

    return (x1, y1, x2, y2)