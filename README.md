# 🩺 Diabetes Prediction App (K-Nearest Neighbors)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetes-prediction-knn.streamlit.app)

Proyek ini adalah aplikasi web interaktif yang dibangun dengan **Streamlit** untuk memprediksi risiko diabetes pada pasien berdasarkan data medis dan gaya hidup.

## 🌟 Fitur Utama

* **Algoritma:** Menggunakan model **K-Nearest Neighbors (KNN)** yang telah dilatih.
* **Preprocessing:** Mengimplementasikan `StandardScaler` dan *One-Hot Encoding* untuk data input baru.
* **Imbalanced Data:** Model dilatih menggunakan teknik **SMOTE** (Synthetic Minority Over-sampling Technique) untuk mengatasi ketidakseimbangan kelas (jumlah kasus diabetes lebih sedikit).

## 🚀 Deployment

### Deployment di Streamlit Cloud

1. Push repository ke GitHub
2. Buka [Streamlit Cloud](https://share.streamlit.io)
3. Klik "New app"
4. Pilih repository, branch, dan file (`app.py`)
5. Klik "Deploy"

### Persyaratan Lokal

Untuk menjalankan aplikasi ini secara lokal, pastikan Anda memiliki Python 3.8+ dan pustaka berikut (`requirements.txt`):

```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi Lokal

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

## 📁 Struktur File

```
.
├── app.py                 # Aplikasi Streamlit utama
├── knn_model.pkl         # Model KNN yang dilatih (pickle)
├── scaler.pkl            # StandardScaler yang disimpan (pickle)
├── requirements.txt      # Dependensi Python
├── .streamlit/
│   └── config.toml      # Konfigurasi Streamlit
├── .gitignore           # File yang diabaikan Git
└── README.md            # Dokumentasi ini
```

## 📊 Input Data

Aplikasi menerima input berikut:

| Fitur | Tipe | Range |
|-------|------|-------|
| Umur (Age) | Numerik | 0.08 - 80 tahun |
| BMI | Numerik | 10.0 - 99.0 |
| HbA1c Level | Numerik | 3.5% - 9.0% |
| Blood Glucose Level | Numerik | 80 - 400 mg/dL |
| Jenis Kelamin | Kategorikal | Female / Male |
| Hipertensi | Biner | Yes (1) / No (0) |
| Penyakit Jantung | Biner | Yes (1) / No (0) |
| Riwayat Merokok | Kategorikal | never, former, current, not current, ever, No Info |

## ⚠️ Disclaimer

Hasil prediksi dari aplikasi ini adalah hasil model machine learning dan **BUKAN** diagnosis medis resmi. Selalu konsultasikan hasil dengan tenaga medis profesional sebelum membuat keputusan medis.

## 🔧 Troubleshooting

**Error: File model atau scaler tidak ditemukan**
- Pastikan file `knn_model.pkl` dan `scaler.pkl` ada di direktori yang sama dengan `app.py`

**Port 8501 sudah digunakan**
```bash
streamlit run app.py --server.port 8502
```

## 📝 Lisensi

Proyek ini bebas digunakan untuk keperluan pendidikan.
