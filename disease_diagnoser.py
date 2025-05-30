import pandas as pd
import joblib

# Load the model
model_path = r"models\model.pkl"

def Load_Model(model_path):
    model = joblib.load(model_path)
    return model 

model = Load_Model(model_path)

# Define the prediction function
def predict_lung_disease(
    GENDER,
    AGE,
    SMOKING,
    YELLOW_FINGERS,
    ANXIETY,
    PEER_PRESSURE,
    CHRONIC_DISEASE,
    FATIGUE,
    ALLERGY,
    WHEEZING,
    ALCOHOL_CONSUMING,
    COUGHING,
    SHORTNESS_OF_BREATH,
    SWALLOWING_DIFFICULTY,
    CHEST_PAIN
):
    classes = ["Non-Cancer", "Cancer"]
    
    # Create input dictionary
    patient_data = {
        "GENDER": [GENDER],
        "AGE": [AGE],
        "SMOKING": [SMOKING],
        "YELLOW_FINGERS": [YELLOW_FINGERS],
        "ANXIETY": [ANXIETY],
        "PEER_PRESSURE": [PEER_PRESSURE],
        "CHRONIC DISEASE": [CHRONIC_DISEASE],
        "FATIGUE ": [FATIGUE],
        "ALLERGY ": [ALLERGY],
        "WHEEZING": [WHEEZING],
        "ALCOHOL CONSUMING": [ALCOHOL_CONSUMING],
        "COUGHING": [COUGHING],
        "SHORTNESS OF BREATH": [SHORTNESS_OF_BREATH],
        "SWALLOWING DIFFICULTY": [SWALLOWING_DIFFICULTY],
        "CHEST PAIN": [CHEST_PAIN]
    }

    # Convert to DataFrame
    df = pd.DataFrame(patient_data)
    
    # Make prediction and probability
    results = model.predict(df)
    prob = model.predict_proba(df)[0]
    
    # Format probabilities
    cancer_prob = round(prob[1] * 100, 2)
    non_cancer_prob = round(prob[0] * 100, 2)
    
    return {
        "prediction": classes[results[0]],
        "cancer_probability (%)": float(cancer_prob),
        "non_cancer_probability (%)": float(non_cancer_prob)
    }

