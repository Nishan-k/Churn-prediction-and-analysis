from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd


# 1. Load the model:
model = joblib.load("./customer_churn_ml/churn_clf_model.pkl")


# 2. Create a base model for the input features:
class Input_features(BaseModel):
    gender: str
    senior_citizen:str
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float




app = FastAPI()

@app.get("/")
def landing_page():
    return {"Heello"}


@app.post("/predict")
def predict_churn(input_features:Input_features):
    try:
        input_data = pd.DataFrame([input_features.model_dump()])
        result = model.predict(input_data)
        pred_prob = model.predict_proba(input_data)
        pred_prob = pred_prob[0][pred_prob.argmax()]
        prediction = result.tolist()[0]
        return {"Prediction": prediction, "Prediction_proba": pred_prob}
    except Exception as e:
        return {'error': str(e)}




