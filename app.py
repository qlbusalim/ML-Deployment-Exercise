import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

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
        gender = st.selectbox("Gender", options=["Male", "Female"])
        pulse_rate = st.number_input("Pulse Rate", min_value=40, max_value=200, value=70)
        systolic_bp = st.number_input("Systolic BP", min_value=60, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
        family_diabetes = st.selectbox("Family History of Diabetes", options=["No", "Yes"])
        hypertensive = st.selectbox("Hypertensive", options=["No", "Yes"])
        family_hypertension = st.selectbox("Family History of Hypertension", options=["No", "Yes"])
        cardiovascular_disease = st.selectbox("Cardiovascular Disease", options=["No", "Yes"])
        stroke = st.selectbox("Stroke", options=["No", "Yes"])

    submitted = st.form_submit_button("Predict")

# PREDICTION
if submitted:
    # Encode categorical variables
    gender_encoded = 0 if gender == "Female" else 1
    family_diabetes_encoded = 0 if family_diabetes == "No" else 1
    hypertensive_encoded = 0 if hypertensive == "No" else 1
    family_hypertension_encoded = 0 if family_hypertension == "No" else 1
    cardiovascular_disease_encoded = 0 if cardiovascular_disease == "No" else 1
    stroke_encoded = 0 if stroke == "No" else 1

    # Create feature array with all 14 features in the correct order
    features = np.array([[
        age,
        gender_encoded,
        pulse_rate,
        systolic_bp,
        diastolic_bp,
        glucose,
        height,
        weight,
        bmi,
        family_diabetes_encoded,
        hypertensive_encoded,
        family_hypertension_encoded,
        cardiovascular_disease_encoded,
        stroke_encoded
    ]])

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.write("---")

    if prediction == 1:
        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style='color:#c0392b;'>⚠️ High Risk of Diabetes</h2>
                <p style='font-size:18px;'>Estimated Probability: <b>{prob:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style='color:#27ae60;'>✔️ Low Risk of Diabetes</h2>
                <p style='font-size:18px;'>Estimated Probability: <b>{prob:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
