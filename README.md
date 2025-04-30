**❤️ Heart Disease Prediction App**

A machine learning–powered web application that predicts the likelihood of heart disease using a trained LightGBM model based on clinical data.


🔗 Live Demo: https://heartdiseasesapp.streamlit.app/
________________________________________
📊 Model Performance (LightGBM on SMOTE Data)
Metric	Score

Accuracy	89.7%

Precision	89.6%
Recall	89.7%

F1 Score	89.6%

ROC AUC Score	94.1%
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
o	Cholesterol-to-Age Ratio
o	MaxHR-to-Age Ratio
o	RestingBP-to-Age Ratio
o	Age Bracket Classification
3.	Data Scaling
Standardized using StandardScaler.
4.	Prediction
Model returns heart disease likelihood with a probability score.
5.	Result Display
Prediction shown along with bar chart visualization.
⚠️ Borderline predictions (45%–55%) trigger a medical caution warning.
________________________________________
🔍 Key Model Insights
-	Cholesterol-Age Ratio and Oldpeak (ST Depression) are the top predictors.
-	ST_Slope_Up is negatively correlated with disease presence.
-	Exercise Angina and RestingBP show strong positive correlation with heart disease.
-	SHAP Visualizations provide transparency in both global and individual predictions.
________________________________________
🚀 Features
✅- Clean and interactive Streamlit UI
✅- Real-time predictions with LightGBM
✅- SHAP visualizations for interpretability
✅- Borderline risk alerts for clinical relevance
✅- Built for medical transparency and early risk screening
________________________________________
🛠️ Tech Stack
-	Machine Learning: scikit-learn, LightGBM, SMOTE
-	Web Application: Streamlit
-	Data Processing: Pandas, NumPy
-	Visualization: Plotly, Matplotlib, SHAP
-	Deployment: Pickle, Streamlit Cloud / Local Hosting

