import streamlit as st
import pandas as pd
import json

# =========================
# CONTROL DETEKSI
# =========================
def set_detection(state):
    with open("control.json", "w") as f:
        json.dump({"run": state}, f)

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# LOGIN DOSEN
# =========================
USERNAME = "dosen"
PASSWORD = "12345"

if "login" not in st.session_state:
    st.session_state.login = False

# =========================
# FORM LOGIN
# =========================
if not st.session_state.login:

    st.title("🔐 Login Dosen")

    user = st.text_input("👤 Username")
    pw = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pw == PASSWORD:
            st.session_state.login = True
            st.success("Login berhasil!")
            st.rerun()
        else:
            st.error("Username atau password salah")

# =========================
# DASHBOARD
# =========================
else:

    st.title("👨‍🏫 Dashboard Monitoring Fokus Mahasiswa")

    # logout
    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.rerun()

    # =========================
    # 🔥 TOMBOL KONTROL (FIX POSISI)
    # =========================
    st.markdown("### 🎛️ Kontrol Deteksi")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("▶️ Mulai Deteksi"):
            set_detection(True)
            st.success("Deteksi AKTIF")

    with col_btn2:
        if st.button("⏹️ Berhenti Deteksi"):
            set_detection(False)
            st.warning("Deteksi DIHENTIKAN")

    # =========================
    # LOAD DATA
    # =========================
    try:
        pd.read_csv("./data_log.csv")

        if df.empty:
            st.warning("Belum ada data")
            st.stop()

        # =========================
        # FILTER KELAS
        # =========================
        st.markdown("### 🏫 Filter Kelas")
        kelas_list = df["kelas"].dropna().unique()
        selected_kelas = st.selectbox("Pilih Kelas", kelas_list)

        df_kelas = df[df["kelas"] == selected_kelas]

        # =========================
        # METRIC
        # =========================
        st.markdown("### 📊 Ringkasan Kelas")

        col1, col2, col3 = st.columns(3)

        total = len(df_kelas)
        fokus = len(df_kelas[df_kelas["status"] == "Fokus"])
        tidak = len(df_kelas[df_kelas["status"] == "Tidak Fokus"])

        persen = (fokus / total * 100) if total > 0 else 0

        col1.metric("Total Data", total)
        col2.metric("Fokus", fokus)
        col3.metric("Tidak Fokus", tidak)

        st.metric("🎯 Persentase Fokus Kelas", f"{persen:.2f}%")

        # =========================
        # GRAFIK
        # =========================
        st.markdown("### 📈 Grafik Fokus")
        st.bar_chart(df_kelas["status"].value_counts())

        # =========================
        # REKAP MAHASISWA
        # =========================
        st.markdown("### 👨‍🎓 Rekap per Mahasiswa")

        rekap = df_kelas.groupby(["nama", "npm"]).agg(
            total_data=("status", "count"),
            fokus=("status", lambda x: (x == "Fokus").sum()),
            tidak=("status", lambda x: (x == "Tidak Fokus").sum())
        ).reset_index()

        rekap["Persentase Fokus (%)"] = (
            rekap["fokus"] / rekap["total_data"] * 100
        )

        rekap["Status Akhir"] = rekap["Persentase Fokus (%)"].apply(
            lambda x: "Fokus" if x >= 50 else "Tidak Fokus"
        )

        st.dataframe(rekap, use_container_width=True)

        # =========================
        # RANKING
        # =========================
        st.markdown("### 🏆 Ranking Mahasiswa")

        ranking = rekap.sort_values(
            by="Persentase Fokus (%)",
            ascending=False
        ).reset_index(drop=True)

        ranking.index += 1
        ranking["Ranking"] = ranking.index

        ranking = ranking[[
            "Ranking", "nama", "npm",
            "Persentase Fokus (%)", "Status Akhir"
        ]]

        st.dataframe(ranking, use_container_width=True)

        # =========================
        # DATA DETAIL
        # =========================
        st.markdown("### 📋 Data Log Detail")
        st.dataframe(df_kelas, use_container_width=True)

        # =========================
        # HAPUS LOG
        # =========================
        st.markdown("### ⚠️ Aksi")

        if st.button("🗑 Hapus Semua Log"):
            df.iloc[0:0].to_csv("data_log.csv", index=False)
            st.success("Log berhasil dihapus!")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
