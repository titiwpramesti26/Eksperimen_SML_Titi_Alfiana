import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# 1. INI ADALAH MESINNYA (FUNGSI UTAMA)
def run_preprocessing(input_path, output_dir):
    print(f"Memulai otomatisasi preprocessing untuk: {input_path}")
    
    # Load Data
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    target_col = 'Outcome' 
    
    # Hapus Duplikat
    df = df.drop_duplicates()
    
    # Separasi X dan y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Imputasi nilai 0 dengan median
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        if col in X.columns:
            X[col] = X[col].replace(0, X[col].median())
            
    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Rekonsiliasi Data
    df_clean = X_scaled_df.copy()
    df_clean[target_col] = y.values
    
    # Simpan Output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data_clean.csv")
    df_clean.to_csv(output_file, index=False)
    
    print(f"Pre-processing Selesai! File disimpan di: {output_file}")


# 2. INI ADALAH TOMBOL START-NYA (YANG SUDAH DISESUAIKAN NAMA FOLDERNYA)
if __name__ == "__main__":
    # Di sini kita pakai folder "diabetes_raw" dan "diabetes_preprocessing" yang sudah kita sepakati
    INPUT = "diabetes_raw/diabetes.csv"
    OUTPUT_DIR = "preprocessing/diabetes_preprocessing"
    
    # Menjalankan mesin dengan input dan output di atas
    run_preprocessing(INPUT, OUTPUT_DIR)