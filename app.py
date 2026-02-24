import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Prediction")

model = joblib.load("model/student_model.pkl")

st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours", 1, 10, 5)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 75)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 17, 25, 20)

gender_val = 0 if gender == "Male" else 1

input_df = pd.DataFrame([[study_hours, attendance, gender_val, age]],
                        columns=["study_hours", "attendance", "gender", "age"])

if st.button("Predict Score"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Exam Score: {round(prediction[0], 2)}")