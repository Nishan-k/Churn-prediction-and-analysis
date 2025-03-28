from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# 1. Load the model:
model = joblib.load("./customer_churn_ml/churn_clf_model.pkl")


app = FastAPI()

@app.get("/")
def landing_page():
    return {"Heello mother fuckers"}


@app.post("/predict")
def predict_churn(input_features):
    df = pd.DataFrame.from_dict(input_features)
    result = model.predict(df)
    return {"Prediction": result}

