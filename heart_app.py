import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ✅ Load Model and Scaler Efficiently
@st.cache_data
def load_model_and_scaler():
    model_path = "LightGBM.pkl"  
    scaler_path = "StandardScalar.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ Model or Scaler file not found! Please upload 'LightGBM.pkl' and 'StandardScalar.pkl'.")
        return None, None

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    return model, scaler

# Load once and use globally
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.stop()

# ✅ Streamlit App Title & Info
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of a patient having heart disease based on clinical parameters.
Simply enter the patient's medical details and get an instant prediction!
""")

# ✅ Input Form UI
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Patient Information")
    
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=50)
    sex = st.radio("Sex", ["Male", "Female"])
    resting_bp = st.slider("Resting Blood Pressure (mmHg)", 80, 180, 120)
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.slider("Max Heart Rate Achieved", 50, 220, 150)
    exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    cholesterol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200)

# ✅ Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
fasting_bs = 1 if fasting_bs == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# ✅ One-hot encoding mappings
chest_pain_mapping = {"Typical Angina": "TA", "Atypical Angina": "ATA", "Non-Anginal Pain": "NAP", "Asymptomatic": "Asym"}
resting_ecg_mapping = {"Normal": "Normal", "ST-T Wave Abnormality": "ST", "Left Ventricular Hypertrophy": "LVH"}
st_slope_mapping = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

chest_pain_types = ['ChestPainType_ATA', 'ChestPainType_NAP']
resting_ecg_types = ['RestingECG_Normal', 'RestingECG_ST']
st_slope_types = ['ST_Slope_Flat']

chest_pain_encoded = {ctype: 0 for ctype in chest_pain_types}
if f"ChestPainType_{chest_pain_mapping[chest_pain_type]}" in chest_pain_types:
    chest_pain_encoded[f"ChestPainType_{chest_pain_mapping[chest_pain_type]}"] = 1

resting_ecg_encoded = {recg: 0 for recg in resting_ecg_types}
if f"RestingECG_{resting_ecg_mapping[resting_ecg]}" in resting_ecg_types:
    resting_ecg_encoded[f"RestingECG_{resting_ecg_mapping[resting_ecg]}"] = 1

st_slope_encoded = {stype: 0 for stype in st_slope_types}
if f"ST_Slope_{st_slope_mapping[st_slope]}" in st_slope_types:
    st_slope_encoded[f"ST_Slope_{st_slope_mapping[st_slope]}"] = 1

# ✅ Feature Engineering
cholesterol_age_ratio = cholesterol / age
max_hr_age_ratio = max_hr / age
resting_bp_age_ratio = resting_bp / age

# ✅ Age Bracket Feature
if age <= 40:
    age_bracket = 0  
elif 40 < age <= 60:
    age_bracket = 1  
else:
    age_bracket = 2  

# ✅ Prepare Input Features
input_features = np.array([[ 
    sex, resting_bp, fasting_bs, exercise_angina, oldpeak, 
    cholesterol_age_ratio, max_hr_age_ratio, resting_bp_age_ratio, age_bracket,
    chest_pain_encoded['ChestPainType_ATA'],
    chest_pain_encoded['ChestPainType_NAP'],
    resting_ecg_encoded['RestingECG_Normal'],
    resting_ecg_encoded['RestingECG_ST'],
    st_slope_encoded['ST_Slope_Flat']
]])

# ✅ Prediction Button
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("❤️ Predict Heart Disease"):
        try:
            input_data_scaled = scaler.transform(input_features)
            probabilities = model.predict_proba(input_data_scaled)[0]
            
            outcome = "Likely to Have Heart Disease" if probabilities[1] >= 0.47 else "Unlikely to Have Heart Disease"
            
            st.success(f"❤️ Prediction: **{outcome}**")
            st.write(f"### 📊 Probability of Heart Disease: {probabilities[1] * 100:.2f}%")
            
            prob_df = pd.DataFrame({
                'Outcome': ['No Heart Disease', 'Heart Disease'],
                'Probability': probabilities * 100
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability',
                         title='Prediction Probabilities',
                         labels={'Probability': 'Probability (%)'},
                         color='Probability',
                         color_continuous_scale='Reds')
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

st.sidebar.header("📌 About")
st.sidebar.info("""
This application predicts the likelihood of heart disease based on patient medical data using **Machine Learning**.

### **Model Information**
- **Algorithm:** LightGBM
- **Trained on:** Heart Disease Dataset  
""")

st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Algorithm Used:** LightGBM
- **Best Accuracy:** 89.7%
- **Precision** 89.6%	
- **Recall** 93.4%	
- **F1 Score** 91.3%
- **ROC AUC Score:** 94.1%
""")

st.subheader("📊 SHAP ANALYSIS")
SHAP_path = "SHAP.png"
if os.path.exists(SHAP_path):
    st.image(Image.open(SHAP_path), caption="Feature Importance", width=700)
