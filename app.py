import streamlit as st

st.set_page_config(page_title="Sistem Deteksi Fokus", layout="wide")

st.title("🎓 Sistem Deteksi Fokus Mahasiswa")
st.write("""
Aplikasi ini digunakan untuk mendeteksi tingkat fokus mahasiswa
menggunakan metode MobileNetV3 dan YOLOv8 secara real-time.
""")

st.info("Silakan pilih menu di sidebar 👈")