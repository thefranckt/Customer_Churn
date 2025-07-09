from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the model
#model = joblib.load("models/logreg_model.pkl")  # Adjust the path if needed

model = joblib.load("../models/logreg_model.pkl")

# Create FastAPI app (named 'api')
api = FastAPI(title="Telco Churn Predictor")

# Define input schema
class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract_Two_year: int
    InternetService_Fiber_optic: int
    OnlineSecurity_No: int
    TechSupport_No: int
    PaperlessBilling: int

# Health check route
@api.get("/")
def health_check():
    return {"status": "API is up and running"}

# Prediction endpoint
@api.post("/predict")
def predict_churn(data: CustomerFeatures):
    input_data = np.array([[
        data.tenure,
        data.MonthlyCharges,
        data.TotalCharges,
        data.Contract_Two_year,
        data.InternetService_Fiber_optic,
        data.OnlineSecurity_No,
        data.TechSupport_No,
        data.PaperlessBilling
    ]])

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": int(prediction),
        "interpretation": "Likely to churn" if prediction == 1 else "Likely to stay"
    }
