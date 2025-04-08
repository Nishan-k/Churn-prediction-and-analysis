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
      You are a **Customer Retention Analyst** AI. Your task is to generate a concise, actionable report explaining why a customer is predicted to churn, based on SHAP values and model predictions. Follow these rules:

      1. **Inputs**: 
         - A dictionary of SHAP values. 
         - A churn prediction (1 for Churn and 0 for Non-Churn).

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
    return system_prompt



def user_prompt(shap_values, predictions):
    user_prompt = f"""
                  Generate a customer churn report using:
                  - SHAP values (var1): f{shap_values}
                  - Churn prediction (var2): f{predictions}

                  Additional context:
                  - Customer tenure: 5 months
                  - Current contract: Month-to-month
                  - Subscription: Basic internet (no add-ons)

                  Focus on the top 3 drivers and prioritize cost-effective interventions.
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

