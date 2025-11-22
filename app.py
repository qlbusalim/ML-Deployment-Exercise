import streamlit as st
import numpy as np
import pandas as pd
import joblib

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# PAGE CONFIG
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
)

# CUSTOM CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 36px !important;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #ecf0f1;
        text-align: center;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='title'>🩺 Diabetes Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Masukkan data pasien untuk memprediksi risiko diabetes</div>", unsafe_allow_html=True)

# INPUT FORM
with st.form("diabetes_form"):
    st.write("### Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)

    with col2:
        bp = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    submitted = st.form_submit_button("Predict")

# PREDICTION
if submitted:
    # Sesuaikan urutan fitur sesuai training model
    features = np.array([[age, bmi, glucose, bp, insulin, skin]])

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.write("---")

    if prediction == 1:
        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style='color:#c0392b;'>⚠️ High Risk of Diabetes</h2>
                <p style='font-size:18px;'>Estimated Probability: <b>{prob:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style='color:#27ae60;'>✔️ Low Risk of Diabetes</h2>
                <p style='font-size:18px;'>Estimated Probability: <b>{prob:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
