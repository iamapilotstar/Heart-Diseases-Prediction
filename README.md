**❤️ HeartGuard: Real-Time ML Tool for Early Heart Disease Detection**

A machine learning–powered web application that predicts the likelihood of heart disease using a trained LightGBM model on clinical indicators. 


🔗 Live Demo: https://heart-diseases-predictions.streamlit.app/

## 💡 The Problem
Healthcare professionals need reliable, transparent tools for early heart disease detection to reduce missed diagnoses and improve patient outcomes through timely intervention.

## 🔧 The Solution
I built a real-time diagnostic application that predicts heart disease risk from clinical data (cholesterol, blood pressure, ECG results) using LightGBM with SHAP-based explanations for transparent medical reasoning.
________________________________________

## 📌 Key Results

- ✅ **Accuracy**: 89.7%  
- ✅ **ROC AUC**: 94.1%  
- ✅ **Precision**: 89.6%  
- ✅ **Recall**: 89.7% (minimizing false negatives)  
- ✅ **F1 Score**: 89.6%  
- ⚠️ Optimized for early detection by prioritizing recall — missing a high-risk patient could be fatal.
________________________________________

📌 Project Overview
Cardiovascular disease is one of the leading global causes of death. This application helps in early screening and risk prediction using key medical indicators:
-	Age & Age Bracket
-	Serum Cholesterol (mg/dL)
- Resting Blood Pressure (mmHg)
-	ST Depression (Oldpeak)
- Chest Pain Type
-	Maximum Heart Rate
-	ECG Results
-	Fasting Blood Sugar
-	Exercise-Induced Angina
-	Slope of the ST Segment
These features are processed and fed into a trained LightGBM model to classify whether the patient is likely or unlikely to have heart disease.
________________________________________
⚙️ How It Works
1.	User Input
Age, cholesterol, ECG results, angina status, ST slope, etc.
2.	Feature Engineering
Cholesterol-to-Age Ratio,
MaxHR-to-Age Ratio,
RestingBP-to-Age Ratio,
Age Bracket Classification,
3.	Data Scaling
Standardized using StandardScaler.
4.	Prediction
Model returns heart disease likelihood with a probability score.
5.	Display the results
   
________________________________________
🔍 Key Model Insights
-	Cholesterol-Age Ratio and Oldpeak (ST Depression) are the top predictors.
-	ST_Slope_Up is negatively correlated with disease presence.
-	Exercise Angina and RestingBP show strong positive correlation with heart disease.
-	SHAP Visualizations provide transparency in both global and individual predictions.
  
________________________________________
🚀 Features
✅- Clean and interactive Streamlit UI, 
✅- Real-time predictions with LightGBM, 
✅- SHAP visualizations for interpretability, 
✅- Built for medical transparency and early risk screening

________________________________________
🛠️ Tech Stack
-	Machine Learning: scikit-learn, LightGBM, SMOTE
-	Web Application: Streamlit
-	Data Processing: Pandas, NumPy
-	Visualization: Plotly, Matplotlib, SHAP
-	Deployment: Pickle, Streamlit Cloud / Local Hosting

## 📁 Folder Structure

Heart-Diseases-Prediction/
│
├── App and Analysis/
│ ├── heart_app.py
│ └── Heart_Diseases_Analysis.ipynb
│
├── Models/
│ ├── LightGBM.pkl
│ └── StandardScaler.pkl
│
├── Images/
│ ├── Confusion Matrix.png
│ └── (SHAP, heatmap, charts, etc.)
│
├── requirements.txt
└── README.md

