import csv
import os
import time
import json

FILE = "data_log.csv"

def init_csv():
    if not os.path.exists(FILE):
        with open(FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Waktu","Nama","NPM","Kelas","Status","Confidence"])

def save_log(nama, npm, kelas, status, conf):
     # 🔥 CEK APAKAH DETEKSI AKTIF
    if not is_detection_active():
        return

    import csv
    from datetime import datetime

    with open("data_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            nama,
            npm,
            kelas,
            status,
            conf
        ])

def is_detection_active():
    try:
        with open("control.json", "r") as f:
            data = json.load(f)
            return data.get("run", False)
    except:
        return False
