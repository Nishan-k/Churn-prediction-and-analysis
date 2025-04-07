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

