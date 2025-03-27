import streamlit as st 
import requests
import json 
import pandas as pd
from PIL import Image
from  customer_churn_ml.data_loader import churn_count
import matplotlib.pyplot as plt
import plotly.express as px

# Navigation Bar:
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.markdown(
    """
    <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #0E4D64;
            color: white;  
        }

        /* Sidebar title */
        [data-testid="stSidebarNav"] {
            font-size: 20px;
            font-weight: bold;
            color: black;
        }

        /* Dropdown menu styling */
        select {
            background-color: #fff;
            color: black;
            border-radius: 5px;
            padding: 5px;
        }

        /* Hover effect for options */
        select option:hover {
            background-color: #d4af37; /* Gold */
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Predict", "📖 Explain", "💡 Recommendations", "ℹ️ About"])
st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")






image = "./images/churn.jpg"
churn_data = churn_count()
# Home Page
if page == "🏠 Home":
     
     st.title("Customer Churn Prediction Model")
     st.write("")
     col1, col2 = st.columns([4, 4])
     with col1:         
        st.write("""
           Customer churn is the loss of customers over time. Predicting churn helps businesses 
                 identify at-risk customers, take proactive retention measures, reduce acquisition costs, 
                 and boost profitability through timely interventions like incentives and personalized services
            """)
     with col2:
         st.image(image)
         st.write("")
         st.write("")

     st.subheader("Current Customer Distribution")
     plt.figure(figsize=(6, 4))
     fig = px.bar(churn_data, x='churn', y='count', color='churn')
     st.plotly_chart(fig)

    
    


# Prediction Page
if page == "📊 Predict":
    st.title("Churn Prediction")

    # User input fields
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.number_input("Tenure (in months)", min_value=1, max_value=72, step=1)
    monthly_spend = st.number_input("Monthly Spend", min_value=0.0, max_value=10000.0, step=0.1)
    
    if st.button("Predict Churn"):
        # Prepare data for API request
        payload = {
            "age": age,
            "contract_type": contract_type,
            "tenure": tenure,
            "monthly_spend": monthly_spend
        }
        
        # Send data to FastAPI for prediction
        response = requests.post(FASTAPI_URL, json=payload)
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.write(f"Churn Prediction: {prediction}")
        else:
            st.error("Error in prediction. Please try again.")
