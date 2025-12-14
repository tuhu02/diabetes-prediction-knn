import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Digunakan untuk deployment

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Diabetes Prediction App (Random Forest)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Memuat Model dan Scaler ---

@st.cache_resource
def load_assets():
    """Memuat model Random Forest dan Scaler yang disimpan."""
    try:
        # Memuat Model Random Forest (Ganti nama file model Anda)
        with open('knn_model.pkl', 'rb') as file: 
            rf_model = pickle.load(file)
        
        # Memuat Scaler
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return rf_model, scaler
    except FileNotFoundError:
        st.error("Error: File model (rf_model.pkl) atau scaler (scaler.pkl) tidak ditemukan.")
        st.stop()

rf_model, scaler = load_assets()

# --- 2. Fungsi Preprocessing ---
# (Fungsi ini tetap sama, karena hanya menangani Scaling dan Encoding)
def preprocess_input(data: dict, scaler: StandardScaler) -> pd.DataFrame:
    """Mengubah data input pengguna menjadi format yang siap diprediksi."""
    
    # 1. Konversi input dictionary menjadi DataFrame
    input_df = pd.DataFrame([data])
    
    # 2. One-Hot Encoding
    input_df['gender_Male'] = np.where(input_df['gender'] == 'Male', 1, 0)
    input_df['smoking_history_current'] = np.where(input_df['smoking_history'] == 'current', 1, 0)
    input_df['smoking_history_ever'] = np.where(input_df['smoking_history'] == 'ever', 1, 0)
    input_df['smoking_history_former'] = np.where(input_df['smoking_history'] == 'former', 1, 0)
    input_df['smoking_history_never'] = np.where(input_df['smoking_history'] == 'never', 1, 0)
    input_df['smoking_history_not current'] = np.where(input_df['smoking_history'] == 'not current', 1, 0)
    
    input_df = input_df.drop(['gender', 'smoking_history'], axis=1)
    
    # 3. Scaling
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # 4. Memastikan urutan kolom sama persis
    final_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 
                  'HbA1c_level', 'blood_glucose_level', 'gender_Male', 
                  'smoking_history_current', 'smoking_history_ever', 
                  'smoking_history_former', 'smoking_history_never', 
                  'smoking_history_not current']
    
    # Menambahkan kolom yang hilang jika kategori tertentu tidak ada dalam input
    for col in final_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    return input_df[final_cols]


# --- 3. Antarmuka Streamlit (UI) ---

st.title("🌳 Aplikasi Prediksi Diabetes (Random Forest)")
st.markdown("Menggunakan model Random Forest dengan penanganan bobot kelas untuk ketidakseimbangan data.")

with st.sidebar:
    st.header("Masukkan Data Pasien")
    
    # ... (Kolom input tetap sama) ...
    age = st.slider("Umur (Age)", min_value=0.08, max_value=80.0, value=40.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=99.0, value=25.0)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.7, format="%.1f")
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=400, value=140, step=1)
    
    st.markdown("---")
    
    gender_input = st.selectbox("Jenis Kelamin (Gender)", options=['Female', 'Male'])
    hypertension = st.selectbox("Hipertensi (Hypertension)", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    heart_disease = st.selectbox("Penyakit Jantung (Heart Disease)", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    smoking_history = st.selectbox("Riwayat Merokok (Smoking History)", 
                                   options=['never', 'No Info', 'former', 'current', 'not current', 'ever'])

    submit_button = st.button("Prediksi")

# --- 4. Logika Prediksi ---

if submit_button:
    
    raw_data = {
        'gender': gender_input, 'age': age, 'hypertension': hypertension, 
        'heart_disease': heart_disease, 'smoking_history': smoking_history, 
        'bmi': bmi, 'HbA1c_level': hba1c, 'blood_glucose_level': glucose
    }
    
    input_processed = preprocess_input(raw_data, scaler)
    
    prediction = rf_model.predict(input_processed)[0]
    prediction_proba = rf_model.predict_proba(input_processed)[0]
    
    st.header("Hasil Prediksi")
    
    # Tampilkan Hasil
    if prediction == 1:
        st.error("⚠️ Pasien Diprediksi **MUNGKIN** Menderita DIABETES (Kelas 1)")
        st.metric(label="Tingkat Keyakinan Prediksi Diabetes", 
                  value=f"{prediction_proba[1]*100:.2f}%")
    else:
        st.success("✅ Pasien Diprediksi **TIDAK** Menderita DIABETES (Kelas 0)")
        st.metric(label="Tingkat Keyakinan Prediksi Non-Diabetes", 
                  value=f"{prediction_proba[0]*100:.2f}%")

    st.subheader("Detail Data yang Digunakan")
    st.json(raw_data)
