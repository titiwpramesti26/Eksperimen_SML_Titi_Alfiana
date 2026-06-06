import requests
import time
import random

# Ini adalah alamat pintu tempat dokter (model ML) kita berjaga nanti
url = "http://localhost:5000/invocations" 

print("Memulai simulasi pengiriman data pasien fiktif...")

# Kita buat perulangan sederhana untuk mengirim 20 data pasien
for i in range(20):
    # Membuat data nilai acak untuk 8 ciri-ciri diabetes
    pasien_fiktif = {
        "dataframe_split": {
            "columns": [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ],
            "data": [[
                random.randint(0, 10),      # Hamil
                random.randint(80, 180),    # Glukosa
                random.randint(60, 90),     # Tekanan Darah
                random.randint(10, 40),     # Ketebalan Kulit
                random.randint(0, 150),     # Insulin
                round(random.uniform(20.0, 35.0), 1), # BMI
                round(random.uniform(0.1, 0.9), 2),   # Silsilah Diabetes
                random.randint(21, 65)      # Umur
            ]]
        }
    }

    try:
        # Mengetuk pintu dan mengirim data pasien
        response = requests.post(url, json=pasien_fiktif)
        print(f"Pasien {i+1} -> Hasil Prediksi Dokter: {response.text.strip()}")
    except Exception as e:
        print(f"Pasien {i+1} -> ⚠ Gagal! Dokter (Model) sepertinya belum menyala.")

    # Jeda 3 detik sebelum pasien berikutnya datang
    time.sleep(3) 

print("Simulasi pengiriman data selesai!")