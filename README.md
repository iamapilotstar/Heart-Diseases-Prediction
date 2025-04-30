**❤️ Heart Disease Prediction App**

This Heart Disease Prediction App uses Machine Learning to predict the likelihood of a patient having heart disease based on key clinical factors.

🔗 Live Demo: https://heartpredictapp.streamlit.app/

📊 Algorithm Used: LightGBM Best Accuracy: 86.9% Precision 89.2% Recall 93.4% F1 Score 91.3% ROC AUC Score: 92.5%

**📌 Project Overview**
Cardiovascular diseases are one of the leading causes of mortality worldwide. This app helps predict heart disease risk based on medical parameters such as:
✔ Age
✔ Cholesterol Levels
✔ Blood Pressure
✔ Chest Pain Type
✔ Heart Rate & ECG Readings
✔ ST Depression & Exercise Induced Angina

**Using these clinical indicators, the model predicts whether the patient is likely or unlikely to have heart disease.
**
📊 How It Works
1️⃣ User Inputs Medical Details → Age, cholesterol, ECG, chest pain, etc.
2️⃣ Data is Preprocessed & Scaled → Standardization using StandardScaler.
3️⃣ Machine Learning Model Predicts the Outcome → Outputs Heart Disease / No Heart Disease.
4️⃣ Displays Probability & Visualization → Bar chart for better insight.

**🔍 Key Insights from Model**
Chest Pain Type (Asymptomatic) and ST Depression are the strongest indicators of heart disease.
Cholesterol & Age Ratio plays a significant role in risk assessment.
Max Heart Rate (HR) and Exercise-Induced Angina show strong correlations with disease presence.

**🚀 Features**
✔ User-friendly Streamlit Interface
✔ Real-time Prediction using a trained LightGBM ✔ Feature Importance Visualization for better understanding
✔ Interactive Sliders & Inputs for easy data entry

**🛠️ Tech Stack**
Machine Learning: Scikit-Learn
Web App Framework: Streamlit
Data Processing: Pandas, NumPy, StandardScaler
Data Visualization: Plotly, Matplotlib, Seaborn
Model Deployment: Pickle (Saved ML Model)
