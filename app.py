import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load('random_forest_model.pkl')

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

# UI Title
st.title("🧠 Cancer Cell Classification")
st.markdown("Predict whether a tumor is **benign** or **malignant** based on diagnostic features.")

# Sidebar Navigation
option = st.sidebar.radio("Choose Input Mode", ['Manual Entry', 'Upload CSV'])

# Prediction function
def predict(features):
    pred = model.predict([features])
    return "Benign" if pred[0] == 1 else "Malignant"

# Manual Entry Mode
if option == 'Manual Entry':
    st.subheader("🔢 Enter Cell Features")

    inputs = []
    for name in feature_names:
        val = st.number_input(f"{name}", min_value=0.0, step=0.01, format="%.2f")
        inputs.append(val)

    if st.button("Predict"):
        result = predict(inputs)
        st.success(f"🎯 Prediction: {result}")

# CSV Upload Mode
else:
    st.subheader("📁 Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check feature match
        if set(feature_names).issubset(df.columns):
            predictions = model.predict(df[feature_names])
            df['Prediction'] = ['Benign' if p == 1 else 'Malignant' for p in predictions]
            st.dataframe(df[['Prediction'] + list(feature_names)])
            st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
        else:
            st.error("❌ CSV must contain all required feature columns.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")
