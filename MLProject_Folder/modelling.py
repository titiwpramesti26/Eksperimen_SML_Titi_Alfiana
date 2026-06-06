import os
import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Inisialisasi Koneksi DagsHub & MLflow Online
USERNAME = "titiwpramesti26"
REPO_NAME = "Eksperimen_SML_Titi_Alfiana"

# Membaca token otomatis dari GitHub Actions jika tersedia
token = os.environ.get("DAGSHUB_TOKEN")

dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True, token=token)
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("Eksperimen_Diabetes_Titi")

# 2. Membaca Data Bersih secara Fleksibel (Mendukung Otomatisasi GitHub Actions)
# Jalur ini akan mencari data bersih hasil dari tahap preprocessing sebelumnya
data_path = os.path.join("preprocessing", "diabetes_preprocessing", "data_clean.csv")

if not os.path.exists(data_path):
    # Jika dijalankan dari dalam subfolder saat pengujian lokal
    data_path = os.path.join("..", "preprocessing", "diabetes_preprocessing", "data_clean.csv")

df = pd.read_csv(data_path)

# 3. Memisahkan Fitur dan Target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 4. Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Menjalankan Otomatisasi Pelatihan Ulang (Re-training)
with mlflow.start_run(run_name="Automated_CI_Retraining"):
    n_estimators = 100
    max_depth = 7
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluasi Hasil Performa
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # MANUAL LOGGING PARAMETERS
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("run_context", "GitHub_Actions_Workflow_CI")
    
    # MANUAL LOGGING METRICS
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # MEMBUAT ARTEFAK TAMBAHAN 1: Gambar Confusion Matrix (.png)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Sehat', 'Diabetes'], yticklabels=['Sehat', 'Diabetes'])
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.title('Confusion Matrix - Automated CI Run')
    
    image_path = "confusion_matrix_ci.png"
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
        
    # MEMBUAT ARTEFAK TAMBAHAN 2: Teks Laporan Klasifikasi (.txt)
    report_text = classification_report(y_test, y_pred, target_names=['Sehat', 'Diabetes'])
    report_path = "classification_report_ci.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
        
    mlflow.log_artifact(report_path)
    if os.path.exists(report_path):
        os.remove(report_path)
        
    # Menyimpan Artefak Model Utama ke Server Online MLflow
    mlflow.sklearn.log_model(model, "model_automated_ci")
    
    print("\n=======================================================")
    print("✓ WORKFLOW CI: RE-TRAINING MODEL BERHASIL DIREKAM ONLINE!")
    print(f"✓ Akurasi Otomatisasi CI: {acc * 100:.2f}%")
    print("=======================================================")