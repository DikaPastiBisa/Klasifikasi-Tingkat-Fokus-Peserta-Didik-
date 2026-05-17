import streamlit as st
import cv2
import time
import torch
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.detection import *
from utils.logging import *

# =========================
# INIT
# =========================
init_csv()

st.set_page_config(layout="wide")

# =========================
# STATE
# =========================
if "run" not in st.session_state:
    st.session_state.run = False

# =========================
# DEVICE STATUS
# =========================
def get_device_status(device_choice):

    if device_choice == "GPU":

        if torch.cuda.is_available():
            return "GPU (Aktif)"

        return "GPU tidak tersedia ❌ (pakai CPU)"

    return "CPU"

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align: center;'>
🎓 Sistem Deteksi Fokus
</h1>

<p style='text-align: center; color: gray;'>
Monitoring fokus mahasiswa secara real-time
</p>
""", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

with col1:

    nama = st.text_input("👤 Nama")

    npm = st.text_input("🆔 NPM")

    kelas = st.text_input("🏫 Kelas")

with col2:

    matkul = st.text_input("📘 Mata Kuliah")

    semester = st.selectbox(
        "📅 Semester",
        ["1","2","3","4","5","6","7","8"]
    )

# =========================
# SETTING
# =========================
st.markdown("### ⚙️ Pengaturan")

col1, col2, col3 = st.columns(3)

with col1:

    model_choice = st.selectbox(
        "🧠 Model",
        ["MobileNetV3", "YOLOv8"]
    )

with col2:

    device = st.selectbox(
        "💻 Device",
        ["CPU", "GPU"]
    )

    device_status = get_device_status(device)

with col3:

    camera_mode = st.selectbox(
        "📷 Kamera",
        ["Web Browser", "OBS Virtual Camera"]
    )

# =========================
# BUTTON
# =========================
colA, colB = st.columns(2)

with colA:

    if st.button("🚀 Mulai Deteksi"):
        st.session_state.run = True

with colB:

    if st.button("🛑 Stop"):
        st.session_state.run = False

# =========================
# LOAD MODEL
# =========================
model = None

if model_choice == "MobileNetV3":

    model = load_mobilenet(
        "models/model_fokus.tflite"
    )

else:

    model = load_yolo(
        "models/model_yolo_best.pt"
    )

# =========================
# LAYOUT
# =========================
col_cam, col_info = st.columns([2,1])

status_placeholder = col_info.empty()

# =========================
# PROCESS FRAME
# =========================
def process_frame(img):

    start_time = time.time()

    label = "-"
    conf = 0

    face_box = detect_face(img)

    if face_box is not None:

        x1, y1, x2, y2 = face_box

        # =========================
        # PRESISI BOX
        # =========================
        padding_x = 20
        padding_y_top = 35
        padding_y_bottom = 15

        x1 += padding_x
        x2 -= padding_x

        y1 += padding_y_top
        y2 -= padding_y_bottom

        # =========================
        # LIMIT AGAR TIDAK ERROR
        # =========================
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # =========================
        # VALIDASI AREA
        # =========================
        if x2 > x1 and y2 > y1:

            face = img[y1:y2, x1:x2]

            if face.size > 0:

                # =========================
                # PREDIKSI
                # =========================
                if model_choice == "MobileNetV3":

                    face_input = preprocess(face)

                    label, conf = predict_mobilenet(
                        model,
                        face_input
                    )

                else:

                    label, conf = predict_yolo(
                        model,
                        face
                    )

                # =========================
                # WARNA
                # =========================
                color = (
                    (0,255,0)
                    if label == "Fokus"
                    else (0,0,255)
                )

                # =========================
                # BOUNDING BOX
                # =========================
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # =========================
                # LABEL
                # =========================
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # =========================
                # CENTER POINT
                # =========================
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5,
                    (0,255,255),
                    -1
                )

                # garis horizontal
                cv2.line(
                    img,
                    (center_x - 20, center_y),
                    (center_x + 20, center_y),
                    (0,255,255),
                    2
                )

                # garis vertikal
                cv2.line(
                    img,
                    (center_x, center_y - 20),
                    (center_x, center_y + 20),
                    (0,255,255),
                    2
                )

    # =========================
    # FPS
    # =========================
    fps = 1 / (time.time() - start_time)

    # =========================
    # INFO BOX
    # =========================
    status_placeholder.markdown(f"""
    ## 📊 Informasi

    - 👤 Nama: **{nama}**
    - 🧠 Model: **{model_choice}**
    - 💻 Device: **{device_status}**
    - 📷 Kamera: **{camera_mode}**

    ---

    - 🎯 Status: **{label}**
    - 📈 Confidence: **{conf:.2f}**

    ---

    - ⚡ FPS: **{fps:.2f}**
    """)

    return img

# =========================
# WEBRTC PROCESSOR
# =========================
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        img = process_frame(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# DETEKSI
# =========================
if st.session_state.run:

    st.success("Deteksi dimulai...")

    # =========================
    # WEB BROWSER CAMERA
    # =========================
    if camera_mode == "Web Browser":

        with col_cam:

            webrtc_streamer(

                key="focus-detection",

                video_processor_factory=VideoProcessor,

                media_stream_constraints={
                    "video": {
                        "width": 480,
                        "height": 360
                    },
                    "audio": False
                },

                rtc_configuration={
                    "iceServers": [
                        {
                            "urls": [
                                "stun:stun.l.google.com:19302"
                            ]
                        }
                    ]
                },

                async_processing=True,
            )

    # =========================
    # OBS CAMERA
    # =========================
    else:

        with col_cam:

            frame_placeholder = st.empty()

            cap = cv2.VideoCapture(1)

            while st.session_state.run:

                ret, frame = cap.read()

                if not ret:
                    st.error("OBS Camera tidak ditemukan")
                    break

                frame = cv2.flip(frame, 1)

                frame = process_frame(frame)

                frame_placeholder.image(
                    frame,
                    channels="BGR"
                )

            cap.release()
