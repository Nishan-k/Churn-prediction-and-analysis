import streamlit as st 
import requests
import json 
import pandas as pd
from PIL import Image
from customer_churn_ml.data_loader import get_churn_distribution, get_churn_count, get_total_customer_counts
import matplotlib.pyplot as plt
import plotly.express as px
import json
import uuid 
from customer_churn_ml.shap_calc import create_clean_shap_dashboard, aggregated_shap_features
from customer_health import customer_health_dashboard, display_dashboard
import joblib
from ui_components.generate_report import get_report
from ui_components.pdf_generator import save_report_as_pdf
import os



st.set_page_config(page_title="Customer Churn Prediction", layout="centered")


def navigate_to_predict():
    st.session_state.page_selection = "📊 Predict"


# Initializing the sessions in Streamlit:
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'input_features' not in st.session_state:
    st.session_state.input_features = None
if 'shap_plot' not in st.session_state:
    st.session_state.shap_plot = None


total_churn_count = get_churn_count()
total_customers = get_total_customer_counts()
baseline_churn_rate = (total_churn_count / total_customers) * 100

contract_mapping = {
    "Month-to-month": 6,
    "One year": 12,
    "Two year": 24
}
##----------------------------------------------------------------------------------##
## CSS Codes:


## Update graph button:
st.markdown("""
    <style>
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: 2px solid #4CAF50;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
            position: relative;
        }
        .stButton>button:hover {
            background-color: #20c942;
            border-color: #c2eced;
            color: white;
        }
        /* Tooltip CSS */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 160px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
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
    churn_data = get_churn_distribution()
    return churn_data


if "page_selection" not in st.session_state:
    st.session_state.page_selection = "🏠 Home"

page = st.sidebar.selectbox("Navigation Menu", ["🏠 Home", "📊 Predict", "📖 Explain", "📑 Generate Report", "ℹ️ About"],
                            key="page_selection")
st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")
st.sidebar.markdown("")




################################
### Landing Page:
################################
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



     if st.button("Update Graph"):  
        chart_key = f"chart_{uuid.uuid4()}"
        with graph_placeholder.container():
            visualize_churn_data(get_data(), chart_key=chart_key)

     st.markdown("</div>", unsafe_allow_html=True)

 

    

################################
### PREDICT PAGE:
################################
if page == "📊 Predict":
    st.title("Churn Prediction")
    st.write("")
    if st.session_state.get('dashboard_displayed', False):
                display_dashboard()
                if st.button("Make New Prediction"):
                    st.session_state.dashboard_displayed = False
                    st.session_state.customer_data = None
                    st.session_state.prediction_result = None
                    st.session_state.res = None
                    st.rerun()

    if not st.session_state.get('dashboard_displayed', False):
        st.subheader("Select Features Or Enter Data to Predict:")
        st.write("")

        # User input fields
        col1, col2 = st.columns([6, 6])

        with col1:
            gender = st.radio("Gender:", ("Male", "Female"))
            senior_citizen = st.radio("Is Senior Citizen?", ["Yes", "No"])
            partner = st.radio("Does the customer have a partner (e.g., spouse or significant other)?",["Yes", "No"])
            dependents = st.radio("Does the customer have dependents (e.g., children, spouse, or family members relying on you)?", ["Yes", "No"])
            tenure = st.number_input("Tenure (In Months):", min_value=1, max_value=100, step=1)
            phone_service = st.radio("Has Phone Service?", ["Yes", "No"])
            multiple_lines = st.radio("Has Multiple Lines?", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service:", ["DSL", "Fibre optic", "No"])
            online_security = st.selectbox("Has intenet security?", ["Yes", "No","No internet service"])
            
        
        with col2:
            online_backup = st.selectbox("Has online backup?", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Has Device Protection?", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Has Tech Support?", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Has Streaming TV?", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Does Customer Stream Movies?", ["Yes", "No", "No internet service"])
            contract = st.radio("Contract Type:", ("One year", "Month-to-month", "Two year"))
            paperless_billing = st.radio("Has Paperless Billing?", ("Yes", "No"))
            payment_method = st.selectbox("Payment Method:", ["Mailed check", "Bank transfer (automatic)", "Electronic check", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charge:", min_value=18.95, max_value=130.0, step=0.1)
            total_charges = st.number_input("Total Charge:", min_value=35.0, max_value=7900.0, step=0.1)


            
        if st.button("Predict Churn"):
            # Prepare data for API request
            input_features = {
                "gender" : gender,
                "senior_citizen" : senior_citizen,
                "partner" : partner,
                "dependents" : dependents,
                "tenure" : tenure,
                "phone_service" : phone_service,
                "multiple_lines" : multiple_lines,
                "internet_service" : internet_service,
                "online_security" : online_security,
                "online_backup" : online_backup,
                "device_protection" : device_protection,
                "tech_support" : tech_support,
                "streaming_tv" : streaming_tv,
                "streaming_movies" : streaming_movies,
                "contract" : contract,
                "paperless_billing" : paperless_billing,
                "payment_method" : payment_method,
                "monthly_charges" : round(monthly_charges, 2),
                "total_charges" : round(total_charges, 2)
            }
            st.session_state.customer_data = input_features
        
    
        
        
            # Send data to FastAPI for prediction
            res = requests.post(url="http://127.0.0.1:8000/predict", json=input_features)
            if res.status_code == 200:
                customer_health_dashboard(res, input_features=input_features)
                
            



if page == "📖 Explain":
    st.title("Churn Explanation")
    if st.session_state.customer_data is None:
        st.warning("Please make a prediction first!")
        if st.button("Go to Prediction Page", on_click=navigate_to_predict):
             pass  
    else:
        st.write("hello")
        

# if page == "💡 Recommendations":
#      st.title("Recommendations")
#      if st.session_state.customer_data is None:
#         st.warning("Please make a prediction first!")
#         if st.button("Go to Prediction Page", on_click=navigate_to_predict):
#              pass  
#      else:
#         st.dataframe(st.session_state.customer_data)


# if page == "📑 Generate Report":
#     st.title("Generate a report for the stakeholder using LLMs")
#     if st.session_state.customer_data is None:
#         st.warning("Please make a prediction first!")
#         if st.button("Go to Prediction Page", on_click=navigate_to_predict):
#              pass  
#     else:
#         prediction = st.session_state.prediction_result
#         customer_data = st.session_state.customer_data
#         aggregated_features = aggregated_shap_features(customer_data=customer_data)
#         st.write(customer_data)
#         st.write("")
#         st.write(aggregated_features)
#         st.write("The Customer will stay" if prediction == 0 else "The customer will leave")
#         st.write("")
#         st.write("")
       
#         if st.button("Generate Report"):
#             report = get_report(shap_values=aggregated_features, predictions=prediction, customer_data=customer_data)
#             st.write(report)

#             # A function to allow the user to download the generated report as a PDF file:
#             pdf_path = save_report_as_pdf(report)
        
#             if pdf_path and os.path.exists(pdf_path):
#                 try:
#                     with open(pdf_path, "rb") as file:
#                         st.download_button(
#                             label="Download as PDF",
#                             data=file,
#                             file_name="Customer_churn_report.pdf",
#                             mime="application/pdf"
#                         )
#                 finally:
#                     try:
#                         os.unlink(pdf_path)  # Delete the temp file
#                     except:
#                         pass
#             else:
#                 st.error("Failed to generate PDF file")

if page == "📑 Generate Report":
    st.title("Generate Report using LLM:")
    if st.session_state.customer_data is None:
        st.warning("Please make a prediction first!")
        if st.button("Go to Prediction Page", on_click=navigate_to_predict):
                pass
    else:
        prediction = st.session_state.prediction_result
        customer_data = st.session_state.customer_data
        aggregated_features = aggregated_shap_features(customer_data=customer_data)

        # Initialize session state
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
            st.session_state.report_content = None
            st.session_state.pdf_path = None

        if st.session_state.customer_data is None:
            st.warning("Please make a prediction first!")
            if st.button("Go to Prediction Page", on_click=navigate_to_predict):
                pass
        else:
            # Show generate button only if report not already generated
            if not st.session_state.report_generated:
                if st.button("Generate Report"):
                    with st.spinner("Generating report..."):
                        # Generate and store in session state
                        st.session_state.report_content = get_report(
                            shap_values=aggregated_features,
                            predictions=prediction,
                            customer_data=customer_data
                        )
                        st.session_state.pdf_path = save_report_as_pdf(st.session_state.report_content)
                        st.session_state.report_generated = True
                    st.rerun()  # Force update to show results

            # Display after generation
            if st.session_state.report_generated:
                st.write(st.session_state.report_content)
                
                if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                    try:
                        with open(st.session_state.pdf_path, "rb") as file:
                            # Use key parameter to maintain button state
                            st.download_button(
                                label="📥 Download as PDF",
                                data=file,
                                file_name="Customer_churn_report.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                    finally:
                        # Cleanup only when leaving the page
                        pass

                # # Add "New Report" button
                # if st.button("🔄 Generate New Report"):
                #     st.session_state.report_generated = False
                #     st.session_state.report_content = None
                #     if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                #         os.unlink(st.session_state.pdf_path)
                #     st.rerun()




################################
### ABOUT PAGE:
################################
if page == "ℹ️ About":
    st.header("Creator: Nishan Karki")





