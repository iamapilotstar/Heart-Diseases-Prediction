**❤️ Heart Disease Prediction App**

A machine learning–powered web application that predicts the likelihood of heart disease based on key clinical indicators using a trained LightGBM model.

**🔗 Live Demo: **


**📊 Model Performance (LightGBM - Tuned on SMOTE Data)**
Test Accuracy: 89.7%

Precision: 89.6%

Recall: 89.7%

F1 Score: 89.6%

ROC AUC: 94.1%

**📌 Project Overview**
Cardiovascular disease is a major global health concern. This application aims to provide early risk screening using key clinical features:

**Age and Age Bracket**, **Serum Cholesterol (mg/dL)**, **Resting Blood Pressure (mmHg)**, **ST Depression Induced by Exercise (Oldpeak)**, **Chest Pain Type** ,**Maximum Heart Rate**, **ECG Results**, **Fasting Blood Sugar**, **Exercise-Induced Angina**, **Slope of ST Segment**

These features are used by a trained model to predict whether a patient is likely or unlikely to have heart disease.

**⚙️ How It Works**
User Input: Age, cholesterol, ECG, angina, ST slope, etc.

**Feature Engineering:**

Cholesterol-to-Age Ratio

MaxHR-to-Age Ratio

RestingBP-to-Age Ratio

Age Bracket Categorization

Data Scaling: Inputs are standardized with StandardScaler.

Prediction: A LightGBM classifier outputs probability of heart disease.

Result Display: Prediction is shown with a probability score and bar chart visualization. Borderline predictions (45%–55%) show a clinical warning.

**🔍 Key Model Insights**
Cholesterol_Age_Ratio and Oldpeak are the most influential features.

ST_Slope_Up is inversely related to heart disease risk.

Exercise Angina and RestingBP show strong positive correlation with disease risk.

SHAP visualizations were used to interpret individual predictions and global feature importance.

**🚀 Features**
✅ Clean and interactive Streamlit UI
✅ Real-time ML prediction using a tuned LightGBM model
✅ Visual explanations via SHAP plots
✅ Warning system for borderline risk cases
✅ Optimized for medical transparency and early screening

**🛠️ Tech Stack**
ML & Modeling: scikit-learn, LightGBM, SMOTE

Web App: Streamlit

Visualization: Plotly, Matplotlib, SHAP

Data Processing: Pandas, NumPy

Deployment: Pickle, Streamlit local/cloud
