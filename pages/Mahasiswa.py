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

col1, col2 = st.columns(2)

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

info_box = col_info.empty()

# =========================
# VIDEO PROCESSOR
# =========================
class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.last_log_time = 0

    def recv(self, frame):

        # =========================
        # FRAME DARI WEBRTC
        # =========================
        img = frame.to_ndarray(format="bgr24")

        # =========================
        # FLIP KAMERA
        # =========================
        img = cv2.flip(img, 1)

        start_time = time.time()

        label = "-"
        conf = 0

        try:

            # =========================
            # DETEKSI WAJAH
            # =========================
            face_box = detect_face(img)

            if face_box is not None:

                x1, y1, x2, y2 = face_box

                # =========================
                # VALIDASI FACE AREA
                # =========================
                if x2 > x1 and y2 > y1:

                    face = img[y1:y2, x1:x2]

                    # =========================
                    # VALIDASI FACE
                    # =========================
                    if face.size > 0:

                        # =========================
                        # PREDIKSI MODEL
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
                        # WARNA BOX
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
                        # SAVE LOG
                        # =========================
                        if (
                            time.time() -
                            self.last_log_time
                        ) > 5:

                            save_log(
                                nama,
                                npm,
                                kelas,
                                label,
                                conf
                            )

                            self.last_log_time = time.time()

            # =========================
            # FPS
            # =========================
            fps = 1 / (
                time.time() - start_time
            )

            # =========================
            # INFO BOX
            # =========================
            info_box.markdown(f"""
            ### 📊 Informasi

            - 👤 Nama: **{nama}**
            - 🧠 Model: **{model_choice}**
            - 💻 Device: **{device_status}**

            ---

            - 🎯 Status: **{label}**
            - 📈 Confidence: **{conf:.2f}**

            ---

            - ⚡ FPS: **{fps:.2f}**
            """)

        except Exception as e:

            cv2.putText(
                img,
                f"Error: {str(e)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# DETEKSI
# =========================
if st.session_state.run:

    st.success("Deteksi dimulai...")

    webrtc_streamer(
        key="focus-detection",

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480
            },
            "audio": False
        },

        async_processing=True,
    )
