import pandas as pd
import os
from datetime import datetime

CSV_FILE = "./data_log.csv"

# =========================
# INIT CSV
# =========================
def init_csv():

    if not os.path.exists(CSV_FILE):

        df = pd.DataFrame(columns=[
            "nama",
            "npm",
            "kelas",
            "status",
            "confidence",
            "timestamp"
        ])

        df.to_csv(
            CSV_FILE,
            index=False
        )

# =========================
# SAVE LOG
# =========================
def save_log(
    nama,
    npm,
    kelas,
    status,
    confidence
):

    try:

        # =========================
        # BACA FILE LAMA
        # =========================
        if os.path.exists(CSV_FILE):

            df = pd.read_csv(CSV_FILE)

        else:

            df = pd.DataFrame(columns=[
                "nama",
                "npm",
                "kelas",
                "status",
                "confidence",
                "timestamp"
            ])

        # =========================
        # DATA BARU
        # =========================
        new_data = pd.DataFrame([{

            "nama": nama,

            "npm": npm,

            "kelas": kelas,

            "status": status,

            "confidence": confidence,

            "timestamp": datetime.now()

        }])

        # =========================
        # GABUNGKAN
        # =========================
        df = pd.concat(
            [df, new_data],
            ignore_index=True
        )

        # =========================
        # SIMPAN
        # =========================
        df.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        print("ERROR SAVE LOG:", e)
