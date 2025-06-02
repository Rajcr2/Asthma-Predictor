import pickle
import pytest
import pandas as pd
from Rule import AsthmaPredictor

# Load Models

@pytest.fixture(scope="module")
def general_model():
    with open("asthma_RB_pred_model_XGB.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def clinical_model():
    with open("strtype_asthma_clinical_model_catboost.pkl", "rb") as f:
        return pickle.load(f)


# Test: General Model (XGB)
def test_general_model_prediction(general_model):
    prediction = general_model.predict(
        age=30,
        bmi=22.5,
        smoking=1,
        wheezing=1,
        nighttime=1,
        sob=0,
        chest=0,
        coughing=0,
        exercise=1
    )

    assert prediction is not None
    assert isinstance(prediction, str)
    print("✅ General model test passed:", prediction)


# Test: Clinical Model (CatBoost)
@pytest.mark.parametrize("input_data", [
    {
        'Age': 45,
        'Gender': 'Male',
        'BMI': 29.5,
        'Smoking': 'Smoker',
        'Wheezing': '0',
        'ShortnessOfBreath': '1',
        'ChestTightness': '0',
        'Coughing': '0',
        'NighttimeSymptoms': '0',
        'ExerciseInduced': '1',
        'FEV1': 3.12,
        'FVC': 5.16
    },
    {
        'Age': 28,
        'Gender': 'Female',
        'BMI': 21.0,
        'Smoking': 'Non-Smoker',
        'Wheezing': '1',
        'ShortnessOfBreath': '1',
        'ChestTightness': '1',
        'Coughing': '1',
        'NighttimeSymptoms': '1',
        'ExerciseInduced': '1',
        'FEV1': 1.75,
        'FVC': 2.1
    },
    {
        'Age': 60,
        'Gender': 'Male',
        'BMI': 31.2,
        'Smoking': 'Smoker',
        'Wheezing': '1',
        'ShortnessOfBreath': '0',
        'ChestTightness': '0',
        'Coughing': '1',
        'NighttimeSymptoms': '1',
        'ExerciseInduced': '0',
        'FEV1': 2.85,
        'FVC': 3.0
    }
])
def test_clinical_model_prediction(clinical_model, input_data):
    df = pd.DataFrame([input_data])
    df = df[clinical_model.feature_names_]

    prediction = clinical_model.predict(df)[0]
    proba = clinical_model.predict_proba(df)[0][1]  # Class 1 = High Risk

    assert prediction in [0, 1], "Prediction should be binary (0 or 1)"
    assert 0.0 <= proba <= 1.0, "Probability must be between 0 and 1"

    if prediction == 1:
        result = f"⚠️ High risk of Asthma. (Confidence: {proba:.2f})"
    else:
        result = f"✅ Low risk of Asthma. (Confidence: {1 - proba:.2f})"

    print("✅ Clinical model test passed:", result)
