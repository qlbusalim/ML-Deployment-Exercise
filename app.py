import streamlit as st
import numpy as np
import pickle
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 1.1rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' exists.")
        return None

model = load_model()

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="title-text">🏥 Prediksi Risiko Diabetes</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Sistem Prediksi Kesehatan Berbasis AI</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Tentang Aplikasi Ini</h3>
        <p>Aplikasi ini menggunakan Machine Learning untuk memprediksi risiko diabetes berdasarkan data kesehatan pasien. 
        Masukkan data pasien untuk mendapatkan prediksi risiko diabetes.</p>
    </div>
""", unsafe_allow_html=True)

# Main section
st.divider()

if model is not None:
    # Input section
    st.subheader("📋 Data Pasien", divider="blue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Informasi Pribadi")
        age = st.number_input(
            "👤 Usia (tahun)",
            min_value=1,
            max_value=120,
            value=30,
            help="Masukkan usia pasien dalam tahun"
        )
        blood_pressure = st.number_input(
            "💓 Tekanan Darah (mmHg)",
            min_value=0.0,
            value=70.0,
            step=0.5,
            help="Masukkan tekanan darah sistolik"
        )
    
    with col2:
        st.markdown("### Metrik Kesehatan")
        glucose = st.number_input(
            "🩸 Kadar Glukosa (mg/dL)",
            min_value=0.0,
            value=100.0,
            step=0.5,
            help="Masukkan kadar glukosa darah"
        )
        bmi = st.number_input(
            "⚖️ BMI",
            min_value=0.0,
            value=20.0,
            step=0.1,
            help="Masukkan Indeks Masa Tubuh"
        )
    
    # Display input summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Usia", f"{int(age)} tahun", "👤")
    with col2:
        st.metric("Glukosa", f"{glucose:.1f} mg/dL", "🩸")
    with col3:
        st.metric("BMI", f"{bmi:.1f}", "⚖️")
    with col4:
        st.metric("Tekanan Darah", f"{blood_pressure:.1f} mmHg", "💓")
    
    st.divider()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Prediksi Diabetes", use_container_width=True, type="primary"):
            with st.spinner("🔄 Sedang menganalisis data..."):
                # Prepare data
                data = np.array([[age, glucose, bmi, blood_pressure]])
                
                # Make prediction
                pred = model.predict(data)[0]
                prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None
                
                st.divider()
                
                # Display results
                if pred == 1:
                    st.markdown(f"""
                        <div class="prediction-box risk-high">
                            <h3>⚠️ HASIL PREDIKSI: BERISIKO TINGGI</h3>
                            <p><strong>Status:</strong> Pasien berisiko diabetes</p>
                            <p><strong>Tingkat Kepercayaan:</strong> {prob*100:.1f}%</p>
                            <hr>
                            <p><strong>Rekomendasi:</strong></p>
                            <ul>
                                <li>Segera konsultasi dengan dokter</li>
                                <li>Lakukan pemeriksaan darah lebih lanjut</li>
                                <li>Terapkan pola hidup sehat</li>
                                <li>Kurangi konsumsi gula dan karbohidrat sederhana</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box risk-low">
                            <h3>✅ HASIL PREDIKSI: RISIKO RENDAH</h3>
                            <p><strong>Status:</strong> Pasien tidak berisiko diabetes</p>
                            <p><strong>Tingkat Kepercayaan:</strong> {(1-prob)*100:.1f}%</p>
                            <hr>
                            <p><strong>Rekomendasi:</strong></p>
                            <ul>
                                <li>Pertahankan pola hidup sehat</li>
                                <li>Lakukan pemeriksaan rutin setiap tahun</li>
                                <li>Olahraga secara teratur (minimal 30 menit/hari)</li>
                                <li>Jaga berat badan ideal</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional visualization
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Analisis Risiko")
                    risk_percentage = (prob * 100) if prob is not None else 0
                    st.progress(risk_percentage / 100, text=f"Tingkat Risiko: {risk_percentage:.1f}%")
                
                with col2:
                    st.subheader("📈 Detail Prediksi")
                    if prob is not None:
                        prediction_data = {
                            "Risiko Diabetes": f"{prob*100:.1f}%",
                            "Tidak Berisiko": f"{(1-prob)*100:.1f}%"
                        }
                        st.bar_chart(prediction_data)
else:
    st.error("❌ Gagal memuat model. Pastikan file 'model.pkl' ada di direktori yang sama.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #999; padding: 20px;'>
        <p>📱 Aplikasi Prediksi Diabetes | v1.0</p>
        <p>Catatan: Prediksi ini hanya untuk tujuan informatif. Untuk diagnosis resmi, konsultasikan dengan dokter profesional.</p>
    </div>
""", unsafe_allow_html=True)