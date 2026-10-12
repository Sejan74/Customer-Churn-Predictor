import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #1F2937;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #374151;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_churn_model.pkl')


model = load_model()

# Header Section
st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("Predict customer retention risk using advanced machine learning models.")

st.divider()

# Layout: 2 Columns (Inputs on left, Results on right)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Customer Profile")

    with st.expander("Account & Financial Details", expanded=True):
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0,
                                        value=float(tenure * monthly_charges), step=1.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    with st.expander("Services & Subscriptions", expanded=False):
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        paperless_val = 1 if paperless == "Yes" else 0

    predict_btn = st.button("Analyze Risk Profile")

with col2:
    st.subheader("🎯 Risk Assessment & Insights")

    # Preprocessing Input Data to match Training Feature Schema
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': 0,
        'Partner_Yes': 1,
        'Dependents_Yes': 0,
        'PhoneService_Yes': 1,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 1 if internet != "No" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'OnlineSecurity_No internet service': 1 if internet == "No" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if internet == "No" else 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 1 if internet == "No" else 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 1 if internet == "No" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if internet == "No" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if internet == "No" else 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaperlessBilling_Yes': paperless_val,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }])

    # Ensure column order matches model input
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    if predict_btn:
        prediction_prob = model.predict_proba(input_data)[0][1]
        risk_percentage = round(prediction_prob * 100, 1)

        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin-bottom: 8px;">Churn Probability</h4>
                <h1 style="color: {'#DC2626' if risk_percentage > 50 else '#059669'}; font-size: 48px; margin: 0;">{risk_percentage}%</h1>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.progress(prediction_prob)

        if risk_percentage > 50:
            st.error("⚠️ High Risk of Churn! This customer is likely to leave the service.")
            st.markdown("""
            **Recommended Retention Actions:**
            * Offer a discounted annual contract plan.
            * Assign a dedicated customer success agent.
            * Provide complimentary upgrades for premium tech support.
            """)
        else:
            st.success("✅ Low Risk. Customer shows strong loyalty indicators.")
            st.markdown("""
            **Recommended Up-sell Actions:**
            * Target for premium plan upgrades.
            * Introduce new value-added add-ons.
            """)
    else:
        st.info(
            "Adjust the customer parameters on the left and click **Analyze Risk Profile** to generate real-time metrics.")