import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("AI Diabetes Risk Prediction System")

st.write("Enter patient health details:")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)

glucose = st.number_input("Glucose", min_value=0, step=1)

blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)

skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)

insulin = st.number_input("Insulin", min_value=0, step=1)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=0, step=1)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
        