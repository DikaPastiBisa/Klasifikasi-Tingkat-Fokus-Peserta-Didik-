import time
import torch
import av
import json

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

@@ -16,6 +17,23 @@

st.set_page_config(layout="wide")

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
# STATE
# =========================
@@ -247,134 +265,147 @@
2


                    # =========================
                    # SAVE LOG KE DOSEN
                    # =========================
                    if get_detection_status():

                        save_log(
                            nama,
                            npm,
                            kelas,
                            label,
                            conf
                        )

# =========================
# FPS
# =========================
fps = 1 / (
time.time() - start_time
)

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

except:
pass

return img

# =========================
# WEBRTC PROCESSOR
# =========================
class VideoProcessor(VideoProcessorBase):

def recv(self, frame):

img = frame.to_ndarray(format="bgr24")

# =========================
# UNMIRROR CAMERA
# =========================
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
# WEB CAMERA
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

st.error(
"OBS Virtual Camera tidak ditemukan"
)

break

# =========================
# UNMIRROR OBS
# =========================
frame = cv2.flip(frame, 1)

frame = process_frame(frame)

frame_placeholder.image(
frame,
channels="BGR"
)

cap.release()
