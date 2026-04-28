# AI-Powered Diabetes Risk Prediction System

## Project Overview
This project predicts whether a patient is at high or low risk of diabetes using Machine Learning.

The system:
- Takes patient health details as input
- Predicts diabetes risk
- Uses Explainable AI (SHAP) to explain predictions
- Provides an interactive web interface using Streamlit

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

---

## Dataset
Dataset used: Pima Indians Diabetes Dataset from Kaggle

Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target:
- Outcome (0 = Low Risk, 1 = High Risk)

---

## Project Workflow

1. Load dataset  
2. Preprocess data  
3. Train machine learning model  
4. Evaluate accuracy  
5. Explain predictions using SHAP  
6. Build Streamlit web application  

---

## Model Accuracy
72%

---

## Files in Project

- load_data.py
- preprocess.py
- train_model.py
- evaluate.py
- explain.py
- app.py
- model.pkl

---

## How to Run

Install dependencies:

pip install pandas scikit-learn xgboost shap streamlit joblib matplotlib

Run application:

python -m streamlit run app.py

---

## Output
The system predicts:

- High Risk of Diabetes
OR
- Low Risk of Diabetes

---

## Future Improvements
- Add wearable health data
- Add personalized recommendations
- Deploy on cloud
- Multi-disease prediction
