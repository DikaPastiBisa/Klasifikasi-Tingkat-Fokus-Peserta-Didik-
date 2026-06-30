import streamlit as st
import cv2
import time
import torch
import av
import json

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.detection import *
from utils.logging import *

# =========================
# INIT
# =========================
init_csv()

st.set_page_config(
    page_title="Deteksi Fokus Mahasiswa",
    layout="wide"
)

# =========================
# CONTROL DETEKSI DOSEN
# =========================
def get_detection_status():

    try:

        with open("control.json", "r") as f:

            data = json.load(f)

            return data.get("run", False)

    except:

        return False


# =========================
# SESSION STATE
# =========================
if "run" not in st.session_state:
    st.session_state.run = False


# =========================
# DEVICE STATUS
# =========================
def get_device_status(device_choice):

    if device_choice == "GPU":

        if torch.cuda.is_available():

            return "GPU Aktif ✅"

        return "GPU Tidak Tersedia ❌ (Menggunakan CPU)"

    return "CPU"


# =========================
# HEADER
# =========================
st.markdown(
"""
<h1 style='text-align:center;'>
🎓 Sistem Deteksi Tingkat Fokus Mahasiswa
</h1>

<p style='text-align:center;color:gray;'>
Monitoring Fokus Mahasiswa Secara Real-Time
</p>
""",
unsafe_allow_html=True
)


# =========================
# INPUT MAHASISWA
# =========================
col1, col2 = st.columns(2)

with col1:

    nama = st.text_input(
        "👤 Nama"
    )

    npm = st.text_input(
        "🆔 NPM"
    )

    kelas = st.text_input(
        "🏫 Kelas"
    )

with col2:

    matkul = st.text_input(
        "📘 Mata Kuliah"
    )

    semester = st.selectbox(
        "📅 Semester",
        [
            "1","2","3","4",
            "5","6","7","8"
        ]
    )


# =========================
# PENGATURAN
# =========================
st.markdown("### ⚙️ Pengaturan")

col1, col2, col3 = st.columns(3)

# =========================
# MODEL
# =========================
with col1:

    model_choice = st.selectbox(
        "🧠 Model",
        [
            "MobileNetV3",
            "YOLOv8"
        ]
    )


# =========================
# DEVICE
# =========================
with col2:

    device = st.selectbox(
        "💻 Device",
        [
            "CPU",
            "GPU"
        ]
    )

    device_status = get_device_status(device)


# =========================
# KAMERA
# =========================
with col3:

    camera_mode = st.selectbox(
        "📷 Kamera",
        [
            "Browser Camera"
        ]
    )


st.info("""
### 📷 Cara Menggunakan Kamera

1. Klik **🚀 Mulai Deteksi**
2. Klik tombol **START**
3. Browser akan meminta izin kamera
4. Pilih salah satu kamera:

- Integrated Camera
- OBS Virtual Camera
- USB Camera
- Logitech Camera

Kemudian klik **Allow**.
""")


# =========================
# BUTTON
# =========================
colA, colB = st.columns(2)

with colA:

    if st.button("🚀 Mulai Deteksi"):

        if (
            nama.strip() == ""
            or npm.strip() == ""
            or kelas.strip() == ""
            or matkul.strip() == ""
        ):

            st.warning(
                "Nama, NPM, Kelas, dan Mata Kuliah wajib diisi!"
            )

        else:

            st.session_state.run = True


with colB:

    if st.button("🛑 Stop"):

        st.session_state.run = False


# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner=True)
def load_model(model_choice):

    if model_choice == "MobileNetV3":

        return load_mobilenet(
            "models/model_fokus.tflite"
        )

    return load_yolo(
        "models/model_yolo_best.pt"
    )


model = load_model(model_choice)


# =========================
# LAYOUT
# =========================
col_cam, col_info = st.columns([2,1])

status_placeholder = col_info.empty()

# =========================
# VIDEO PROCESSOR
# =========================
class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.last_detection_time = 0
        self.last_ui_update = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        start_time = time.time()

        label = "-"
        conf = 0.0

        try:

            # =========================
            # DETEKSI WAJAH
            # =========================
            face_box = detect_face(img)

            if face_box is not None:

                x1, y1, x2, y2 = face_box

                # =========================
                # HITUNG TITIK TENGAH WAJAH
                # =========================
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # =========================
                # UKURAN BOUNDING BOX
                # =========================
                box_size = 180

                x1 = center_x - box_size // 2
                x2 = center_x + box_size // 2

                y1 = center_y - box_size // 2
                y2 = center_y + box_size // 2

                # =========================
                # BATASI AREA GAMBAR
                # =========================
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                # =========================
                # VALIDASI AREA WAJAH
                # =========================
                if x2 > x1 and y2 > y1:

                    face = img[y1:y2, x1:x2]

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
                        # WARNA BOUNDING BOX
                        # =========================
                        color = (
                            (0, 255, 0)
                            if label == "Fokus"
                            else (0, 0, 255)
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
                            f"{label} ({conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                        # =========================
                        # SAVE LOG
                        # =========================
                        current_time = time.time()

                        if (
                            current_time - self.last_detection_time >= 5
                        ):

                            if get_detection_status():

                                if (
                                    nama.strip() != ""
                                    and npm.strip() != ""
                                    and kelas.strip() != ""
                                ):

                                    try:

                                        save_log(
                                            nama.strip(),
                                            npm.strip(),
                                            kelas.strip(),
                                            label,
                                            round(float(conf), 2)
                                        )

                                    except Exception as e:

                                        print("Save Log Error :", e)

                                    self.last_detection_time = current_time

                    else:

                        label = "Tidak Ada Wajah"
                        conf = 0.0

                else:

                    label = "Tidak Ada Wajah"
                    conf = 0.0

            else:

                label = "Tidak Ada Wajah"
                conf = 0.0

            # =========================
            # FPS
            # =========================
            fps = 1 / max(
                (time.time() - start_time),
                0.001
            )

            # =========================
            # UPDATE INFO (1 DETIK SEKALI)
            # =========================
            if time.time() - self.last_ui_update >= 1:

                status_placeholder.markdown(f"""
                ## 📊 Informasi

                👤 **Nama**
                - {nama}

                🧠 **Model**
                - {model_choice}

                💻 **Device**
                - {device_status}

                📷 **Kamera**
                - {camera_mode}

                ---

                🎯 **Status**
                - **{label}**

                📈 **Confidence**
                - **{conf:.2f}**

                ⚡ **FPS**
                - **{fps:.2f}**
                """)

                self.last_ui_update = time.time()

        except Exception as e:

            print("VideoProcessor Error :", e)

            cv2.putText(
                img,
                "ERROR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # =========================
        # RETURN FRAME
        # =========================
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# DETEKSI
# =========================
if st.session_state.run:

    st.success("Deteksi dimulai...")

    if camera_mode == "Browser Camera":

        with col_cam:

            webrtc_ctx = webrtc_streamer(

                key="focus-detection",

                video_processor_factory=VideoProcessor,

                media_stream_constraints={

                    "video":{

                        "width":{
                            "ideal":640
                        },

                        "height":{
                            "ideal":480
                        },

                        "frameRate":{
                            "ideal":15
                        },

                        "facingMode":"user"

                    },

                    "audio":False

                },

                rtc_configuration={

                    "iceServers":[

                        {
                            "urls":[
                                "stun:stun.l.google.com:19302",
                                "stun:stun1.l.google.com:19302",
                                "stun:stun2.l.google.com:19302",
                                "stun:stun3.l.google.com:19302",
                                "stun:stun4.l.google.com:19302"
                            ]
                        }

                    ]

                },

                async_processing=True,

            )

            if webrtc_ctx.state.playing:

                st.success("📷 Kamera berhasil aktif")

            else:

                st.info("Silakan klik START kemudian pilih kamera pada browser.")
