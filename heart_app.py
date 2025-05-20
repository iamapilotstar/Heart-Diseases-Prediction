import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# Page Configuration
st.set_page_config(page_title="❤️ Heart Disease Predictor & Insights", layout="wide")
st.title("❤️ Heart Disease Prediction & Model Insights")
st.markdown("🡸 **Use the sidebar on the left to switch between prediction and model insights.**")

st.markdown("🡻 **Scroll or swipe down in the Model Insights section below the images to view detailed explanations.**")

st.markdown("⚠️ *This is a portfolio project. Please do not use it for real medical diagnosis or clinical decisions. Always consult a licensed medical professional.*")

view_option = st.sidebar.radio("Choose View", ["🔍 Predict", "📊 Model Insights"])

# ---------- 🔍 PREDICTION SECTION ----------
if view_option == "🔍 Predict":
    @st.cache_data
    def load_model_and_scaler():
        try:
            with open("LightGBM.pkl", "rb") as f:
                model = pickle.load(f)
            with open("StandardScalar.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except:
            return None, None

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("⚠️ Model or Scaler file not found!")
        st.stop()

    st.markdown("This app predicts the likelihood of a patient having heart disease based on clinical data.")

    # Input Form
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🧾 Patient Information")
        age = st.number_input("Age (years)", 1, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"])
        resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=180, value=120)
        chest_pain_type = st.selectbox("Chest Pain Type", [
            "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"
        ])
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
        resting_ecg = st.selectbox("Resting ECG Results", [
            "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"
        ])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
        exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f")
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", [
            "Upsloping", "Flat", "Downsloping"
        ])
        cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

    # Encode inputs and feature engineering...
    sex = 1 if sex == "Male" else 0
    fasting_bs = 1 if fasting_bs == "Yes" else 0
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    chest_map = {
        "Typical Angina": "TA", "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP", "Asymptomatic": "Asym"
    }
    ecg_map = {
        "Normal": "Normal", "ST-T Wave Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH"
    }
    slope_map = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

    chest_encoded = {'ChestPainType_ATA': 0, 'ChestPainType_NAP': 0}
    if f"ChestPainType_{chest_map[chest_pain_type]}" in chest_encoded:
        chest_encoded[f"ChestPainType_{chest_map[chest_pain_type]}"] = 1

    ecg_encoded = {'RestingECG_Normal': 0, 'RestingECG_ST': 0}
    if f"RestingECG_{ecg_map[resting_ecg]}" in ecg_encoded:
        ecg_encoded[f"RestingECG_{ecg_map[resting_ecg]}"] = 1

    slope_encoded = {'ST_Slope_Flat': 0}
    if slope_map[st_slope] == 'Flat':
        slope_encoded['ST_Slope_Flat'] = 1

    # Derived Features
    cholesterol_age_ratio = cholesterol / age
    max_hr_age_ratio = max_hr / age
    resting_bp_age_ratio = resting_bp / age
    age_bracket = 0 if age <= 40 else 1 if age <= 60 else 2

    # Assemble feature vector
    input_features = np.array([[ 
        sex,
        resting_bp,
        fasting_bs,
        exercise_angina,
        oldpeak,
        cholesterol_age_ratio,
        max_hr_age_ratio,
        resting_bp_age_ratio,
        age_bracket,
        chest_encoded['ChestPainType_ATA'],
        chest_encoded['ChestPainType_NAP'],
        ecg_encoded['RestingECG_Normal'],
        ecg_encoded['RestingECG_ST'],
        slope_encoded['ST_Slope_Flat']
    ]])

    with col1:
        if st.button("❤️ Predict Heart Disease"):
            try:
                scaled = scaler.transform(input_features)
                probs = model.predict_proba(scaled)[0]
                label = (
                    "Likely to Have Heart Disease" if probs[1] >= 0.47
                    else "Unlikely to Have Heart Disease"
                )
                st.success(f"❤️ Prediction: **{label}**")
                st.write(f"### 📊 Probability of Heart Disease: `{probs[1]*100:.2f}%`")

                prob_df = pd.DataFrame({
                    'Outcome': ['No Heart Disease', 'Heart Disease'],
                    'Probability': probs * 100
                })
                fig = px.bar(
                    prob_df,
                    x='Outcome',
                    y='Probability',
                    title='Prediction Probabilities',
                    labels={'Probability': 'Probability (%)'},
                    color='Probability',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

elif view_option == "📊 Model Insights":
    st.markdown("### 📈 Deep Dive: How the Model Learns from Clinical Data")

    image_paths = {
        "SHAP Summary": "SHAP.png",
        "MaxHR vs Age": "Age Vs Heart.png",
        "Oldpeak Comparison": "Age vs Old Peak.png",
        "Resting BP Trend": "Blood Pressure.png",
        "ECG Results": "ECG.png",
        "Feature Importance": "feature_importance.png",
        "Confusion Matrix": "Confusion Matrix.png",
        "Waterfall": "Waterfall_shap.png",
        "Heatmap": "Heatmap.png",
        "Distributions": "Distribution.png",
        "Class Distribution": 'Class Distribution.png'
    }

    tab_labels = list(image_paths.keys())
    tabs = st.tabs(tab_labels)

    for tab, tab_key in zip(tabs, tab_labels):
        with tab:
            st.subheader(tab_key)
            img_file = image_paths[tab_key]
            if os.path.exists(img_file):
                st.image(Image.open(img_file), width=700)
            else:
                st.error(f"⚠️ Image file not found: {img_file}")

            if tab_key == "SHAP Summary":
                st.markdown("""
                - **SHAP – BEESWARM PLOT**
                - SHAP (SHapley Additive exPlanations) was used to interpret how each feature contributes to the model’s output.
                - For instance, high cholesterol and typical angina significantly reduce the risk, while asymptomatic chest pain and oldpeak values tend to increase it.
                - This ensures the model is not a black box and provides clinically valid justifications.
                """)
            
            
            elif tab_key == "MaxHR vs Age":
                st.markdown("""
                - **AGE VS HEART RATE**
                - Analyzes how maximum heart rate achieved changes with age, stratified by heart disease presence.
                - A clear downward trend shows patients with heart disease generally have lower HR and higher age which aids in identifying high-risk cases effectively.
                - Useful for both feature engineering and medical insight.
                """)

            elif tab_key == "Oldpeak Comparison":
                st.markdown("""
                - **INTERPRETING EXERCISE-INDUCED ST DEPRESSION (OLDPEAK)**
                - **Definition**: Oldpeak measures the depth of ST-segment depression during peak exercise compared to rest (in mm).
                - **Significance**: Greater depression indicates myocardial ischemia – insufficient blood flow to cardiac muscle under load.
                - **Observation**: The bar chart shows patients diagnosed with heart disease have, on average, 3x higher Oldpeak values than those without disease.
                - **Clinical Cutoff**: An Oldpeak >2.0 mm is often used to flag significant ischemia; in our data, the diseased group centers around 1.2–1.3 mm, underscoring even mild depressions as predictive.
                - **Model Role**: Ranked among top features, Oldpeak helps the model capture latent ischemic risk beyond resting ECG.
                """)

            elif tab_key == "Resting BP Trend":
                st.markdown("""
                - **AGE VS BP**
                - Explores how resting blood pressure correlates with age and disease status.
                - The upward trend indicates higher BP in older age groups, especially for patients with heart disease.
                """)

            elif tab_key == "ECG Results":
                st.markdown("""
                - **ECG VS PERCENTAGES**
                - Compares ECG outcomes (Normal, ST wave abnormality, LVH) with heart disease incidence.
                - Reveals that left ventricular hypertrophy (LVH) is most associated with positive heart disease diagnoses.
                - Helped ensure better model convergence.
                """)

            elif tab_key == "Feature Importance":
                st.markdown("""
                - **FEATURE IMPORTANCE**
                - This plot highlights the top 20 most influential features ranked by importance in the LightGBM model.
                - ChestPainType, Age, and Cholesterol levels emerged as critical predictors of heart disease.
                - This visualization guided feature selection and reinforced clinical correlations.
                """)


            elif tab_key == "Heatmap":
                st.markdown("""
                - **EVALUATING HEATMAP**
                - This correlation heatmap displays the pairwise relationships between all features in the dataset.
                - While it identified potential multicollinearity (e.g., between Cholesterol and Oldpeak), it was not addressed in this project since tree-based models like LightGBM are not sensitive to multicollinearity.
                - In scenarios involving linear models, common techniques to handle multicollinearity include dropping redundant features or using Variance Inflation Factor (VIF) to filter out variables exceeding a set threshold.
                - Beyond technical checks, this plot also enhances EDA storytelling by visually capturing how clinical indicators relate to each other, supporting feature interpretation and domain understanding.
                """)

            elif tab_key == "Confusion Matrix":
                st.markdown("""
                - **EVALUATING CLASSIFICATION PERFORMANCE**
                - **True Positives (94)**: Correctly identified as diseased — strong sensitivity (recall ~92%).
                - **True Negatives (71)**: Correctly identified as healthy — good specificity (~86%).
                - **False Positives (11)**: Healthy patients flagged — acceptable in screening to avoid missing cases.
                - **False Negatives (8)**: Diseased missed — minimized to reduce risk of undiagnosed pathology.
                - **Clinical Balance**: Prioritizing low false negatives ensures high-risk patients receive follow-up; slight increase in false positives is tolerable for safety.
                - **Overall Accuracy**: (71+94)/(71+11+8+94) ~ 89%.
                """)

            elif tab_key == "Waterfall":
                st.markdown("""
                - **SHAP – WATERFALL PLOT**
                - This SHAP waterfall plot explains a single prediction made by the model by breaking down how each feature increases or decreases the probability of predicting heart disease.
                - Features like ExerciseAngina and ST_Slope_Up increased the risk prediction, while features like Fasting Blood Sugar (FastingBS) and Cholesterol-to-Age Ratio helped to lower it.
                - The final prediction value (f(x) = 2.96) indicates a strong lean towards class 1 (Heart Disease) for this specific individual.
                - This level of explainability helps build trust in the model’s decision-making, especially in sensitive healthcare applications where clarity is critical.
                """)

            elif tab_key == "Distributions":
                st.markdown("""
                - **DISTRIBUTION OF FEATURES**
                - This visualization shows the distribution of key numerical features such as Cholesterol, Resting Blood Pressure, Oldpeak, and others.
                - It helps identify skewness, outliers, and overall data distribution patterns, which are crucial for making informed preprocessing decisions.
                - For instance, highly skewed features may require log transformation, while the presence of outliers can influence the choice between mean vs. median imputation for handling missing values.
                - Such insights ensure the data is well-prepared and model-friendly, improving both performance and interpretability.
                """)

            elif tab_key == "Class Distribution":
                st.markdown("""
                - **CLASS DISTRIBUTION**
                - This pie chart displays the class distribution of the dataset:
                - 🟥 44.7% without heart disease
                - 🟩 55.3% with heart disease
                - The slightly imbalanced distribution emphasizes the need for a robust model that performs well across both classes, particularly the positive class (heart disease) to avoid under-diagnosis.
                """)

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
