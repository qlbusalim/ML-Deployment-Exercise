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
    .high-risk {
        background-color: #fadbd8;
        border-left: 5px solid #c0392b;
    }
    .low-risk {
        background-color: #d5f4e6;
        border-left: 5px solid #27ae60;
    }
    .risk-meter {
        width: 100%;
        height: 30px;
        background-color: #ecf0f1;
        border-radius: 15px;
        overflow: hidden;
        margin: 15px 0;
    }
    .risk-fill-low {
        height: 100%;
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        border-radius: 15px;
    }
    .risk-fill-high {
        height: 100%;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        border-radius: 15px;
    }
    .recommendation {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #3498db;
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
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", options=["Female", "Male"])
        pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=200, value=70)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=500, value=100)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = weight / (height ** 2)  # Auto-calculate BMI
        st.metric("BMI (Auto-calculated)", f"{bmi:.2f}")
        family_diabetes = st.selectbox("Family History of Diabetes", options=["No", "Yes"])
        hypertensive = st.selectbox("Hypertensive", options=["No", "Yes"])
        family_hypertension = st.selectbox("Family History of Hypertension", options=["No", "Yes"])
        cardiovascular_disease = st.selectbox("Cardiovascular Disease", options=["No", "Yes"])
        stroke = st.selectbox("Stroke", options=["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

# PREDICTION
if submitted:
    # Encode categorical variables (matching training data encoding)
    gender_encoded = 1 if gender == "Male" else 0
    family_diabetes_encoded = 1 if family_diabetes == "Yes" else 0
    hypertensive_encoded = 1 if hypertensive == "Yes" else 0
    family_hypertension_encoded = 1 if family_hypertension == "Yes" else 0
    cardiovascular_disease_encoded = 1 if cardiovascular_disease == "Yes" else 0
    stroke_encoded = 1 if stroke == "Yes" else 0

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

    # Get prediction and probability
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]  # Probability of having diabetes (class 1)

    st.write("---")
    st.write("### 📊 Prediction Results")

    # Determine risk level based on probability
    if prob < 0.3:
        risk_level = "🟢 Low Risk"
        risk_color = "low-risk"
        risk_fill_class = "risk-fill-low"
        risk_percentage = prob * 100
    elif prob < 0.6:
        risk_level = "🟡 Moderate Risk"
        risk_color = "moderate-risk"
        risk_fill_class = "risk-fill-moderate"
        risk_percentage = prob * 100
    else:
        risk_level = "🔴 High Risk"
        risk_color = "high-risk"
        risk_fill_class = "risk-fill-high"
        risk_percentage = prob * 100

    # Display risk level
    st.markdown(
        f"""
        <div class='result-box {risk_color}'>
            <h2 style='margin: 0;'>{risk_level}</h2>
            <p style='font-size: 16px; margin: 10px 0; color: #555;'>Diabetes Risk Probability</p>
            <h1 style='margin: 10px 0; color: #2c3e50;'>{prob:.1%}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk meter visualization
    st.markdown(
        f"""
        <div class='risk-meter'>
            <div class='{risk_fill_class}' style='width: {min(risk_percentage, 100)}%;'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display risk factors
    st.write("### ⚠️ Risk Factors Identified:")
    risk_factors = []
    
    if glucose > 126:
        risk_factors.append(f"• High Glucose Level: {glucose} mg/dL (Normal: < 100 mg/dL)")
    if bmi >= 30:
        risk_factors.append(f"• Obesity: BMI {bmi:.2f} (Normal: < 25)")
    if age >= 45:
        risk_factors.append(f"• Age: {age} years (45+ is higher risk)")
    if systolic_bp > 140 or diastolic_bp > 90:
        risk_factors.append(f"• High Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg (Normal: < 120/80)")
    if family_diabetes == "Yes":
        risk_factors.append("• Family History of Diabetes")
    if hypertensive == "Yes":
        risk_factors.append("• Hypertension Diagnosis")
    if cardiovascular_disease == "Yes":
        risk_factors.append("• Cardiovascular Disease")
    if stroke == "Yes":
        risk_factors.append("• Previous Stroke")

    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("✅ No major risk factors identified")

    # Provide recommendations
    st.write("### 💡 Recommendations:")
    
    if prob >= 0.6:
        recommendations = [
            "🏥 Schedule an appointment with an endocrinologist immediately",
            "🧪 Get a fasting blood glucose test and HbA1c test",
            "📋 Comprehensive diabetes screening is strongly recommended",
            "💊 Follow doctor's advice regarding medication if needed",
            "🏃 Begin a supervised exercise program (30 min/day, 5 days/week)",
            "🥗 Consult with a dietitian for a diabetes prevention diet"
        ]
    elif prob >= 0.3:
        recommendations = [
            "🔍 Schedule a check-up with your healthcare provider",
            "🧪 Undergo diabetes screening tests",
            "🏃 Increase physical activity to at least 30 minutes daily",
            "🥗 Reduce sugar and refined carbohydrate intake",
            "⚖️ Maintain a healthy weight",
            "👨‍⚕️ Monitor blood glucose levels regularly"
        ]
    else:
        recommendations = [
            "✅ Continue regular health check-ups",
            "🏃 Maintain an active lifestyle (30+ min exercise, 5 days/week)",
            "🥗 Eat a balanced diet rich in vegetables and whole grains",
            "⚖️ Keep a healthy weight",
            "🧪 Regular health screening tests as per age guidelines",
            "🚫 Avoid smoking and limit alcohol consumption"
        ]

    for i, rec in enumerate(recommendations, 1):
        st.markdown(
            f"""
            <div class='recommendation'>
                {rec}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Disclaimer
    st.info(
        "⚠️ **Disclaimer**: This prediction is for informational purposes only and should not be considered as a medical diagnosis. "
        "Please consult with a qualified healthcare professional for accurate diagnosis and treatment recommendations."
    )
