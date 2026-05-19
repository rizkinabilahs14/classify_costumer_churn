import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# 2. Fungsi untuk memuat model dan komponen
@st.cache_resource
def load_components():
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/best_model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, encoders, model

try:
    scaler, encoders, model = load_components()

    st.title("📱 Customer Churn Classification")
    st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi potensi Churn.")

    # 3. Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=['Female', 'Male'])
            tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=100, value=20)
            usage_freq = st.number_input("Usage Frequency (Per Bulan)", min_value=0, max_value=50, value=15)
            support_calls = st.number_input("Support Calls (6 Bulan Terakhir)", min_value=0, max_value=20, value=2)

        with col2:
            pay_delay = st.number_input("Payment Delay (Hari)", min_value=0, max_value=30, value=5)
            sub_type = st.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
            contract = st.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'])

        submit = st.form_submit_button("Prediksi Sekarang")

    if submit:
        # 4. Data Preparation
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Tenure': tenure,
            'Usage Frequency': usage_freq,
            'Support Calls': support_calls,
            'Payment Delay': pay_delay,
            'Subscription Type': sub_type,
            'Contract Length': contract
        }])

        # 5. Preprocessing (Encoding & Scaling)
        for col, le in encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col].astype(str))
        
        input_scaled = scaler.transform(input_data)

        # 6. Prediksi
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)

        # 7. Tampilkan Hasil
        st.divider()
        if prediction[0] == 1:
            st.error(f"🚨 Hasil: Pelanggan berpotensi CHURN (Probabilitas: {prob[0][1]:.2%})")
        else:
            st.success(f"✅ Hasil: Pelanggan tetap SETIA (Probabilitas: {prob[0][0]:.2%})")

except FileNotFoundError:
    st.error("Model atau komponen (scaler/encoder) tidak ditemukan di folder 'models/'. Pastikan Anda sudah mengekspornya.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
