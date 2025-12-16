import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Diabetes Prediction App (KNN)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Memuat Model dan Scaler ---

@st.cache_resource
def load_assets():
    """Memuat model KNN dan Scaler yang disimpan."""
    try:
        # Tentukan path file relatif terhadap direktori aplikasi
        base_path = Path(__file__).parent
        model_path = base_path / 'knnmodel.pkl'
        scaler_path = base_path / 'scaler.pkl'
        
        # Memastikan file ada
        if not model_path.exists() or not scaler_path.exists():
            st.error(f"Error: File model atau scaler tidak ditemukan di {base_path}")
            st.stop()
        
        # Memuat Model KNN
        with open(model_path, 'rb') as file:
            knn_model = pickle.load(file)
        
        # Memuat Scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
        return knn_model, scaler
    except Exception as e:
        st.error(f"Error memuat model atau scaler: {str(e)}")
        st.stop()

knn_model, scaler = load_assets()

# --- 2. Fungsi Preprocessing (DIJAMIN SAMA dengan Saat Pelatihan BARU) ---

def preprocess_input(data: dict, scaler: StandardScaler) -> pd.DataFrame:
    """Mengubah data input pengguna menjadi format yang siap diprediksi (encoding & scaling)."""
    
    # 1. Konversi input dictionary menjadi DataFrame
    input_df = pd.DataFrame([data])
    
    # 2. One-Hot Encoding
    
    # Gender (Male menjadi 1, Female menjadi 0)
    input_df['gender_Male'] = np.where(input_df['gender'] == 'Male', 1, 0)
    
    # Smoking History (Jika 'No Info' adalah baseline, maka semua kolom ini akan 0)
    input_df['smoking_history_current'] = np.where(input_df['smoking_history'] == 'current', 1, 0)
    input_df['smoking_history_ever'] = np.where(input_df['smoking_history'] == 'ever', 1, 0)
    input_df['smoking_history_former'] = np.where(input_df['smoking_history'] == 'former', 1, 0)
    input_df['smoking_history_never'] = np.where(input_df['smoking_history'] == 'never', 1, 0)
    input_df['smoking_history_not current'] = np.where(input_df['smoking_history'] == 'not current', 1, 0)
    
    # Hapus kolom kategorikal asli
    input_df = input_df.drop(['gender', 'smoking_history'], axis=1)
    
    # Memastikan semua kolom ada (Termasuk yang biner/boolean, karena scaler dilatih pada semua fitur)
    final_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 
                  'HbA1c_level', 'blood_glucose_level', 'gender_Male', 
                  'smoking_history_current', 'smoking_history_ever', 
                  'smoking_history_former', 'smoking_history_never', 
                  'smoking_history_not current']
    
    # PENTING: Gunakan hanya kolom yang dibutuhkan, lalu pastikan tipe datanya benar
    input_df = input_df[final_cols].astype('float64') # Mengubah tipe data ke float64
    
    # 3. Scaling (Sekarang kita scale SEMUA KOLOM karena scaler BARU dilatih pada semua kolom)
    # Scaler yang baru dilatih pada notebook sudah mencakup semua fitur
    input_processed_scaled = scaler.transform(input_df)
    
    return pd.DataFrame(input_processed_scaled, columns=final_cols)


# --- 3. Antarmuka Streamlit (UI) ---

st.title("🏥 Aplikasi Prediksi Diabetes")
st.markdown("Gunakan model K-Nearest Neighbors (KNN) yang dilatih menggunakan data yang di-SMOTE.")

with st.sidebar:
    st.header("Masukkan Data Pasien")
    
    # Kolom Numerik
    age = st.slider("Umur (Age)", min_value=0.08, max_value=80.0, value=40.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=99.0, value=25.0)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, format="%.1f")
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=400, value=140, step=1)
    
    st.markdown("---")
    
    # Kolom Biner/Kategorikal
    gender_input = st.selectbox("Jenis Kelamin (Gender)", options=['Female', 'Male'])
    hypertension = st.selectbox("Hipertensi (Hypertension)", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    heart_disease = st.selectbox("Penyakit Jantung (Heart Disease)", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    smoking_history = st.selectbox("Riwayat Merokok (Smoking History)", 
                                   options=['never', 'No Info', 'former', 'current', 'not current', 'ever'])

    submit_button = st.button("🔮 Prediksi", use_container_width=True)

# --- 4. Logika Prediksi ---

if submit_button:
    
    # 1. Kumpulkan data input
    raw_data = {
        'gender': gender_input,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }
    
    # 2. Preprocess
    try:
        input_processed = preprocess_input(raw_data, scaler)
    except Exception as e:
        st.error(f"Gagal memproses input: {e}")
        st.stop()
        
    # 3. Prediksi dan Probabilitas
    prediction = knn_model.predict(input_processed)[0]
    prediction_proba = knn_model.predict_proba(input_processed)[0]
    
    st.header("Hasil Prediksi")
    
    # 4. Tampilkan Hasil
    if prediction == 1:
        st.error("⚠️ Pasien Diprediksi **MUNGKIN** Menderita DIABETES (Kelas 1)")
        st.metric(label="Tingkat Keyakinan Prediksi Diabetes", 
                  value=f"{prediction_proba[1]*100:.2f}%")
        st.caption("Hasil ini adalah prediksi model dan harus dikonfirmasi oleh tenaga medis profesional.")
    else:
        st.success("✅ Pasien Diprediksi **TIDAK** Menderita DIABETES (Kelas 0)")
        st.metric(label="Tingkat Keyakinan Prediksi Non-Diabetes", 
                  value=f"{prediction_proba[0]*100:.2f}%")
        st.caption("Hasil menunjukkan risiko rendah berdasarkan model, namun pemeriksaan rutin tetap dianjurkan.")

    st.subheader("Detail Data yang Digunakan (Scaled)")
    st.dataframe(input_processed)
