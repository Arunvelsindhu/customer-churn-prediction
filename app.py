import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/churn_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
}

/* Constrain and frame the whole app */
.block-container {
    max-width: 900px;
    padding: 2.5rem 2.5rem 3rem 2.5rem;
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(19,26,43,0.6), rgba(10,14,23,0.6));
    box-shadow: 0 0 40px rgba(0, 212, 255, 0.08);
    margin-top: 2rem;
}

/* Card look for each input section */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background-color: rgba(19, 26, 43, 0.4) !important;
}

.stButton>button {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    border: 1px solid #00d4ff;
    border-radius: 8px;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.4);
}

.stButton>button:hover {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ChurnSense AI")
st.caption("AI-powered churn prediction for smarter retention decisions")
st.write("Enter customer details below to predict the likelihood of churn.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("👤 Customer Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    with st.container(border=True):
        st.subheader("💳 Billing & Services")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

st.divider()

if st.button("Predict Churn", type="primary", use_container_width=True):

    service_flags = [online_security, online_backup, device_protection,
                      tech_support, streaming_tv, streaming_movies]
    num_services = sum(1 for s in service_flags if s == "Yes")

    customer_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'NumServices': num_services
    }

    customer_df = pd.DataFrame([customer_dict])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **Prediction: Likely to Churn**")
    else:
        st.success(f"✅ **Prediction: Likely to Stay**")

    st.metric("Churn Probability", f"{probability:.1%}")
    st.progress(probability)