import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IEEE-CIS Fraud Detection Models Data.csv")
    return df

df = load_data()

# ---------------------------------
# App Title
# ---------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Fraud Detection Analysis App")
st.markdown("This app analyzes fraud risk using transaction data.")

st.divider()

# ---------------------------------
# Dataset Preview
# ---------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.divider()

# ---------------------------------
# Dashboard Metrics
# ---------------------------------
st.subheader("📊 Dashboard Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    st.metric("Average Fraud Risk", round(df["isFraud"].mean(), 4))

st.divider()

# ---------------------------------
# Fraud Score Distribution Graph
# ---------------------------------
st.subheader("📈 Fraud Score Distribution")

fig, ax = plt.subplots()
ax.hist(df["isFraud"], bins=30)
ax.set_xlabel("Fraud Score")
ax.set_ylabel("Number of Transactions")
ax.set_title("Fraud Score Distribution")

st.pyplot(fig)

st.divider()

# ---------------------------------
# Transaction Fraud Risk Checker
# ---------------------------------
st.subheader("🔍 Check Transaction Fraud Risk")

transaction_id = st.number_input(
    "Enter Transaction ID",
    min_value=int(df["TransactionID"].min()),
    max_value=int(df["TransactionID"].max()),
    value=int(df["TransactionID"].iloc[0])
)

if st.button("Check Fraud Risk"):

    result = df[df["TransactionID"] == transaction_id]

    if result.empty:
        st.warning("Transaction ID not found")
    else:
        fraud_score = float(result["isFraud"].values[0])

        st.write(f"### Fraud Score: **{fraud_score:.4f}**")

        if fraud_score > 0.5:
            st.error("🚨 High Risk Transaction")
            st.write("This transaction has a **high chance of being fraud**.")
        else:
            st.success("✅ Low Risk Transaction")
            st.write("This transaction has a **low chance of being fraud**.")

st.divider()

# ---------------------------------
# Footer
# ---------------------------------
st.caption("🚀 Fraud Detection Project | Machine Learning & Data Analysis")
