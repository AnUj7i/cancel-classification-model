import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df, cancer

df, cancer = load_data()

# Title
st.title("🧬 Cancer Cell Classification")
st.markdown("Explore the Breast Cancer dataset using Streamlit and Plotly")

# Basic Info
st.subheader("📌 Dataset Overview")
st.write(df.head())
st.write(df.describe())

# Target distribution
st.subheader("🎯 Target Class Distribution")
fig1 = px.histogram(df, x="target", color="target",
                    color_discrete_map={0: 'red', 1: 'green'},
                    labels={"target": "Cancer Type (0 = Malignant, 1 = Benign)"})
st.plotly_chart(fig1)

# Correlation heatmap
st.subheader("🔗 Correlation Heatmap")
fig2, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig2)

# Feature relationships
st.subheader("📈 Feature Comparison: Mean Radius vs Mean Texture")
fig3 = px.scatter(df, x="mean radius", y="mean texture", color=df["target"].astype(str),
                  labels={"color": "Cancer Type"})
st.plotly_chart(fig3)

# Select feature distribution
st.subheader("📌 Feature Distribution Viewer")
feature = st.selectbox("Choose a feature", df.columns[:-1])
fig4 = px.histogram(df, x=feature, color=df['target'].astype(str),
                    nbins=30, barmode='overlay')
st.plotly_chart(fig4)

st.markdown("---")
st.markdown("Made with ❤️ using Scikit-learn + Streamlit")
