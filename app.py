import streamlit as st
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

st.title("Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi risiko diabetes.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=20.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

# Add more inputs here if your model requires additional features

if st.button("Prediksi Diabetes"):
    data = np.array([[age, glucose, bmi, blood_pressure]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error(f"Hasil: Pasien berisiko diabetes. Probabilitas: {prob:.2f}" if prob is not None else "Hasil: Pasien berisiko diabetes.")
    else:
        st.success(f"Hasil: Pasien tidak berisiko diabetes. Probabilitas: {prob:.2f}" if prob is not None else "Hasil: Pasien tidak berisiko diabetes.")
