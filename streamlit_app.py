import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")


@st.cache_resource
def load_files():
    model = joblib.load("churn_model.joblib")
    scaler = joblib.load("scaler.joblib")
    columns = joblib.load("columns.joblib")
    return model, scaler, columns


model, scaler, columns = load_files()

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a telecom customer's details and click the button to see "
    "whether they are likely to leave."
)

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
    monthly = st.number_input(
        "Monthly charges ($)", min_value=18.0, max_value=120.0,
        value=70.0, step=0.5,
    )
    contract = st.selectbox(
        "Contract type", ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    if internet == "No":
        tech = "No internet service"
        st.selectbox("Tech support", [tech], disabled=True)
    else:
        tech = st.selectbox("Tech support", ["No", "Yes"])

if st.button("Predict"):
    raw = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TechSupport": tech,
    }])
    row = pd.get_dummies(raw, dtype=int).reindex(columns=columns, fill_value=0)
    row[["tenure", "MonthlyCharges"]] = scaler.transform(
        row[["tenure", "MonthlyCharges"]]
    )
    prob = model.predict_proba(row)[0][1]

    if prob >= 0.5:
        st.error("Likely to leave")
    else:
        st.success("Likely to stay")
    st.metric("Churn risk score", f"{prob:.0%}")
    st.caption(
        "The score is a risk score, not an exact probability, because the "
        "model was tuned to catch as many leavers as possible (recall 79%). "
        "About half of the customers it flags will not actually leave."
    )