import streamlit as st
import numpy as np
import pickle
from Rule import AsthmaPredictor  # Prediction interface
import pandas as pd

# Load both models
with open("asthma_RB_pred_model_XGB.pkl", "rb") as f:
    general_predictor = pickle.load(f)

with open("strtype_asthma_clinical_model_catboost.pkl", "rb") as f:
    clinical_predictor = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Asthma Prediction", layout="centered")
st.title("🫁 Asthma Risk Predictor")

user_type = st.sidebar.radio("Select User Type:", ["General User", "Clinical User"])

# -------------------- GENERAL USER PAGE --------------------
if user_type == "General User":
    st.header("General User Assessment")
    st.markdown("Provide your basic health details below:")

    age = st.slider("Age", 0, 100, 25)
    gender = st.radio("Gender", ["Male", "Female"])
    bmi = st.slider("BMI", 10, 50, 22)
    smoking = st.radio("Smoking Status", ["Non-Smoker", "Smoker"])
    wheezing = st.checkbox("Wheezing")
    short_breath = st.checkbox("Shortness of Breath")
    chest_tightness = st.checkbox("Chest Tightness")
    coughing = st.checkbox("Coughing")
    nighttime_symptoms = st.checkbox("Nighttime Symptoms")
    exercise_induced = st.checkbox("Exercise-Induced Symptoms")

    gender_val = 0 if gender == "Male" else 1
    smoking_val = 0 if smoking == "Non-Smoker" else 1

    if st.button("Predict Asthma Risk"):
        result = general_predictor.predict(
            age=int(age),
            bmi=float(bmi),
            smoking=int(smoking_val),
            wheezing=int(wheezing),
            nighttime=int(nighttime_symptoms),
            sob=int(short_breath),
            chest=int(chest_tightness),
            coughing=int(coughing),
            exercise=int(exercise_induced)
        )
        
        if "High risk" in result:
            st.markdown(f"```text\n{result}\n```")  # Use code block style for neat multi-line formatting
        else:
            st.markdown(f"```text\n{result}\n```")

# -------------------- CLINICAL USER PAGE --------------------
elif user_type == "Clinical User":
    st.header("Clinical Assessment")
    st.markdown("Provide detailed clinical parameters:")

    age = st.slider("Age", 0, 100, 25)
    gender = st.radio("Gender", ["Male", "Female"], key="c_gender")
    bmi = st.slider("BMI", 10, 50, 22, key="c_bmi")
    smoking = st.radio("Smoking Status", ["Non-Smoker", "Smoker"], key="c_smoke")
    wheezing = st.checkbox("Wheezing", key="c_wheeze")
    short_breath = st.checkbox("Shortness of Breath", key="c_sob")
    chest_tightness = st.checkbox("Chest Tightness", key="c_chest")
    coughing = st.checkbox("Coughing", key="c_cough")
    nighttime_symptoms = st.checkbox("Nighttime Symptoms", key="c_night")
    exercise_induced = st.checkbox("Exercise-Induced Symptoms", key="c_exercise")

    # Clinical-specific inputs
    fev1 = st.number_input("FEV1 (Forced Expiratory Volume in 1s)", min_value=0.0, format="%.2f")
    fvc = st.number_input("FVC (Forced Vital Capacity)", min_value=0.0, format="%.2f")

    # Calculate and display ratio
    def calculate_ratio(fev1, fvc):
        if fvc > 0:
            return round(fev1 / fvc, 2)
        return 0.0

    ratio = calculate_ratio(fev1, fvc)
    st.markdown(f"**FEV1/FVC Ratio:** `{ratio}`")

    # Convert categorical values to string (as per model training)
    gender_val = gender  # Already string: "Male" or "Female"
    smoking_val = smoking  # Already string: "Smoker" or "Non-Smoker"

    # Checkbox values to string "0" or "1"
    input_dict = {
        'Age': int(age),
        'Gender': gender_val,
        'BMI': float(bmi),
        'Smoking': smoking_val,
        'Wheezing': str(int(wheezing)),
        'ShortnessOfBreath': str(int(short_breath)),
        'ChestTightness': str(int(chest_tightness)),
        'Coughing': str(int(coughing)),
        'NighttimeSymptoms': str(int(nighttime_symptoms)),
        'ExerciseInduced': str(int(exercise_induced)),
        'FEV1': float(fev1),
        'FVC': float(fvc)
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Clinical Asthma Risk Prediction"):
        input_df = input_df[clinical_predictor.feature_names_]
        prediction = clinical_predictor.predict(input_df)[0]
        proba = clinical_predictor.predict_proba(input_df)[0][1]  # class 1 = High Risk

        if prediction == 1:
            result = f"⚠️ High risk of Asthma. (Confidence: {proba:.2f})"
        else:
            result = f"✅ Low risk of Asthma. (Confidence: {1 - proba:.2f})"

        # Display in monospace code block using markdown
        st.markdown(f"```text\n{result}\n```")
