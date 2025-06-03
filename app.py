import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier # Meskipun model diload, ini baik untuk referensi
# Muat model
model_filename = 'gradient_boosting_model.pkl'
try:
    gb_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"File model '{model_filename}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Simpan daftar kolom yang digunakan saat training (penting untuk one-hot encoding)
# Anda bisa mendapatkan ini dari notebook: X_train_rs.columns
# Sebaiknya simpan daftar kolom ini secara eksplisit atau dari file
# Berdasarkan notebook:
# Index(['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
#        'avg_glucose_level', 'bmi', 'work_type_Govt_job',
#        'work_type_Private', 'work_type_Self-employed', 'work_type_children',
#        'Residence_type_Rural', 'Residence_type_Urban',
#        'smoking_status_Unknown', 'smoking_status_formerly smoked',
#        'smoking_status_never smoked', 'smoking_status_smokes'],
#       dtype='object')
# Urutan ini SANGAT PENTING
expected_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi', 'work_type_Govt_job',
    'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'smoking_status_smokes'
]

st.title("Prediksi Stroke 🧠")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan stroke.")

# Input untuk fitur-fitur
# Kolom numerik dasar
age = st.number_input("Usia (Age)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
avg_glucose_level = st.number_input("Rata-rata Level Glukosa (Average Glucose Level)", min_value=0.0, value=100.0)
bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0)

# Kolom biner yang dikonversi manual di notebook
gender_options = ['Male', 'Female'] # Sesuai data asli sebelum konversi
gender_input = st.selectbox("Jenis Kelamin (Gender)", gender_options)
# Konversi di backend: 0 if Male, 1 if Female (sesuai notebook: dataset['gender'] = [0 if i != 'Female' else 1 for i in dataset['gender']])
gender = 1 if gender_input == 'Female' else 0

hypertension_options = ['Tidak (No)', 'Ya (Yes)']
hypertension_input = st.radio("Hipertensi (Hypertension)", hypertension_options)
hypertension = 1 if hypertension_input == 'Ya (Yes)' else 0

heart_disease_options = ['Tidak (No)', 'Ya (Yes)']
heart_disease_input = st.radio("Penyakit Jantung (Heart Disease)", heart_disease_options)
heart_disease = 1 if heart_disease_input == 'Ya (Yes)' else 0

ever_married_options = ['Belum (No)', 'Sudah (Yes)']
ever_married_input = st.selectbox("Status Pernikahan (Ever Married)", ever_married_options)
# Konversi di backend: 0 if No, 1 if Yes (sesuai notebook: dataset['ever_married'] = [0 if i !='Yes' else 1 for i in dataset['ever_married']])
ever_married = 1 if ever_married_input == 'Sudah (Yes)' else 0

# Kolom yang di-one-hot encode (minta input asli, lalu proses)
work_type_options = ['Govt_job', 'Private', 'Self-employed', 'children'] # 'Never_worked' jika ada
work_type_input = st.selectbox("Jenis Pekerjaan (Work Type)", work_type_options)

residence_type_options = ['Rural', 'Urban']
residence_type_input = st.selectbox("Jenis Tempat Tinggal (Residence Type)", residence_type_options)

smoking_status_options = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
smoking_status_input = st.selectbox("Status Merokok (Smoking Status)", smoking_status_options)

if st.button("Prediksi"):
    # Buat dictionary dari input
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        # Inisialisasi semua kolom one-hot encoding dengan False (atau 0)
        'work_type_Govt_job': False,
        'work_type_Private': False,
        'work_type_Self-employed': False,
        'work_type_children': False,
        'Residence_type_Rural': False,
        'Residence_type_Urban': False,
        'smoking_status_Unknown': False,
        'smoking_status_formerly smoked': False,
        'smoking_status_never smoked': False,
        'smoking_status_smokes': False
    }

    # Set kolom one-hot yang sesuai menjadi True berdasarkan input
    if work_type_input in work_type_options: # Cek jika ada 'Never_worked', dll.
        input_data[f'work_type_{work_type_input}'] = True
    
    if residence_type_input in residence_type_options:
        input_data[f'Residence_type_{residence_type_input}'] = True

    if smoking_status_input in smoking_status_options:
        input_data[f'smoking_status_{smoking_status_input}'] = True

    # Buat DataFrame dari input_data dengan urutan kolom yang benar
    input_df = pd.DataFrame([input_data], columns=expected_columns)
    
    # Konversi tipe data boolean ke int jika model dilatih dengan int (get_dummies defaultnya bool, lalu dikonversi saat operasi numerik)
    for col in input_df.columns:
        if input_df[col].dtype == 'bool':
            input_df[col] = input_df[col].astype(int)

    st.subheader("Data Input yang Diproses:")
    st.write(input_df)

# Lakukan prediksi
    try:
        prediction = gb_model.predict(input_df)
        prediction_proba = gb_model.predict_proba(input_df)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.error("Pasien diprediksi BERISIKO terkena stroke.")
            st.write(f"Probabilitas Stroke: {prediction_proba[0][1]*100:.2f}%")
            st.write(f"Probabilitas Tidak Stroke: {prediction_proba[0][0]*100:.2f}%")
        else:
            st.success("Pasien diprediksi TIDAK BERISIKO terkena stroke.")
            st.write(f"Probabilitas Tidak Stroke: {prediction_proba[0][0]*100:.2f}%")
            st.write(f"Probabilitas Stroke: {prediction_proba[0][1]*100:.2f}%")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.error("Pastikan semua input sudah benar dan model kompatibel.")
