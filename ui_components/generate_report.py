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

system_prompt = """
You are a **Customer Retention Analyst** AI. Your task is to generate a concise, actionable report explaining why a customer is predicted to churn, based on SHAP values and model predictions. Follow these rules:

1. **Inputs**: 
   - A dictionary of SHAP values (var1: {feature: impact}). 
   - A churn prediction (var2: True/False).

2. **Output Structure**:
   - **Title**: "Customer Churn Risk Report"
   - **Prediction**: Highlight churn risk (e.g., "High Risk (85%)" or "Low Risk (12%)").
   - **Top Drivers**: List 3-5 most impactful features with plain-English explanations.
   - **Business Interpretation**: Link features to real-world behavior (e.g., "No tech support → frequent unresolved complaints").
   - **Recommendations**: Suggest 2-3 data-backed actions.

3. **Style**:
   - Avoid jargon. Use bullet points.
   - Quantify impact (e.g., "Contract type increased risk by 15%").
   - Ignore features with near-zero SHAP values (<0.01).
   - Never invent facts outside the SHAP data.
"""

user_prompt = f"""
Generate a customer churn report using:
- SHAP values (var1): {shap_vals}
- Churn prediction (var2): {prediction}

Additional context:
- Customer tenure: 5 months
- Current contract: Month-to-month
- Subscription: Basic internet (no add-ons)

Focus on the top 3 drivers and prioritize cost-effective interventions.
"""