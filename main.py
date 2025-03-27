import streamlit as st 
import requests
import json 
import pandas as pd
from PIL import Image
from  customer_churn_ml.data_loader import churn_count
import matplotlib.pyplot as plt
import plotly.express as px
import uuid 


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")


##----------------------------------------------------------------------------------##
## CSS Codes:


## Update graph button:
st.markdown("""
    <style>
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%; /* Ensure the container takes full height */
        }
        .stButton>button {
            background-color: #4CAF50; /* Green */
            color: white;
            border: 2px solid #4CAF50;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #20c942; 
            border-color: #c2eced;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# Navigation Bar:

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

##----------------------------------------------------------------------------------##

image = "./images/churn.jpg"

def visualize_churn_data(churn_data, chart_key):
    custom_colors = {'Yes': '#fc4903', 'No': '#03fc94'} 
    st.subheader("Current Customer Distribution")
    fig = px.bar(get_data(), x='churn', y='count', color='churn', color_discrete_map=custom_colors)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def get_data():
    churn_data = churn_count()
    return churn_data





page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Predict", "📖 Explain", "💡 Recommendations", "ℹ️ About"])
st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")




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
     

     st.write("")
     graph_placeholder = st.empty()
     chart_key = f"chart_{uuid.uuid4()}"
     with graph_placeholder.container():
        visualize_churn_data(get_data(), chart_key=chart_key)
     
     st.markdown('<div class="center-container">', unsafe_allow_html=True)
     if st.button("Update Graph"):
        chart_key = f"chart_{uuid.uuid4()}"

        with graph_placeholder.container():
            visualize_churn_data(get_data(), chart_key=chart_key) 
     st.markdown('</div>', unsafe_allow_html=True)
    
     st.write("")
     st.subheader("Current Data")
     st.dataframe(get_data())
     


    


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
