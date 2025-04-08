import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import joblib
import streamlit as st 






load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("API Key looks good so far.")
else:
    print("Check the API key, there is some issue with it.")

MODEL = 'gpt-4o-mini'
openai = OpenAI()

def system_prompt():
    system_prompt = """
    You are a **Customer Retention Analyst** AI. Your task is to generate a concise, actionable report explaining customer churn risk, based on SHAP values and model predictions. Follow these rules:

    1. **Inputs**: 
       - A dictionary of SHAP values where positive values increase churn risk and negative values decrease it.
       - A binary churn prediction (1 = High Risk of Churn, 0 = Low Risk of Non-Churn).

    2. **Output Structure**:
       - **Title**: "Customer Churn Risk Report"
       - **Prediction**: State "High Risk (Predicted to Churn)" for prediction=1 or "Low Risk (Predicted to Retain)" for prediction=0.
       - **Top Drivers**: List the 3-5 most impactful features based on absolute SHAP values. For each feature:
          * Indicate whether it increases risk (positive SHAP) or decreases risk (negative SHAP)
          * Quantify the impact (e.g., "increases churn risk by approximately X%")
       - **Business Interpretation**: Explain what each top driver means in business terms.
       - **Recommendations**: Suggest 2-3 specific, actionable interventions based on the top drivers.

    3. **Style**:
       - Use clear, non-technical language and bullet points.
       - Correctly interpret SHAP values: Positive SHAP = increases churn probability; Negative SHAP = decreases churn probability.
       - Focus only on features with significant SHAP values (absolute value > 0.01).
       - Base all insights strictly on the provided SHAP data and contextual information.
    """
    return system_prompt


def user_prompt(shap_values, predictions):
    # Format the prediction in a clearer way
    prediction_text = "1 (Will Churn)" if predictions == 1 else "0 (Will Not Churn)"
    
    # Sort SHAP values by absolute magnitude to identify most important features
    sorted_shap = {k: v for k, v in sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    user_prompt = f"""
    Generate a customer churn report using:
    - SHAP values: {sorted_shap}
    - Churn prediction: {prediction_text}

    Additional context:
    - Customer tenure: 5 months
    - Current contract: Month-to-month
    - Subscription: Basic internet (no add-ons)

    Important notes for interpretation:
    1. In the provided SHAP values, POSITIVE values INCREASE churn risk, while NEGATIVE values DECREASE churn risk.
    2. Focus on the top 3 drivers with the highest absolute SHAP values.
    3. When interpreting the prediction: '1' means high risk (will churn), '0' means low risk (won't churn).
    4. Express SHAP values as percentages (e.g., a SHAP value of 0.05 = 5% impact on prediction).
    5. Provide recommendations that are specific and actionable based on the identified drivers.
    """
    return user_prompt

def get_report(shap_values, predictions):
    system = system_prompt()
    user = user_prompt(shap_values=shap_values, predictions=predictions)
    input_data = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=input_data
    )

    return response.choices[0].message.content

