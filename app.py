from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page Configuration with wide layout for a dashboard feel
st.set_page_config(
    page_title="Medical Insurance Cost Predictor", 
    page_icon="🏥", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI, card containers, and clean buttons
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15);
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = Path(__file__).parent / "insurance_cleaned.csv"

@st.cache_resource
def train_model():
    """Train the Multiple Linear Regression model."""
    df = pd.read_csv(DATA_FILE)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X.columns, y.quantile(0.50), y.quantile(0.75)

def bmi_category(b):
    if b < 18.5:
        return "Underweight 🔵"
    if b < 25:
        return "Normal 🟢"
    if b < 30:
        return "Overweight 🟠"
    return "Obese 🔴"

if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75 = train_model()

# ---------------- Interface ----------------
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Provide the patient details below to calculate an accurate estimation of yearly insurance expenses using Machine Learning.")
st.divider()

# Input Form Layout in Columns
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("### 👤 Personal Metrics")
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of Dependents / Children", min_value=0, max_value=5, value=0)

with col2:
    st.markdown("### 📋 Lifestyle & Demographics")
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
    region = st.selectbox("Residential Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown(f"**Current BMI Status:** {bmi_category(bmi)}")
st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("Calculate Insurance Cost", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
    }])[columns]

    cost = max(model.predict(input_data)[0], 0)

    if cost < q50:
        level = "Low Risk / Budget Tier"
    elif cost < q75:
        level = "Moderate Standard Tier"
    else:
        level = "High Risk Premium Tier"

    st.divider()
    st.subheader("🎯 Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Estimated Yearly Cost", value=f"${cost:,.2f}")
    with res_col2:
        st.metric(label="Risk Assessment Level", value=level)

    if cost < q50:
        st.success(f"The estimated yearly insurance expense is calculated at **${cost:,.2f}** (Low Tier).")
    elif cost < q75:
        st.warning(f"The estimated yearly insurance expense is calculated at **${cost:,.2f}** (Moderate Tier).")
    else:
        st.error(f"The estimated yearly insurance expense is calculated at **${cost:,.2f}** (High Risk Tier).")

    with st.expander("🔍 View Raw Model Input Data"):
        st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("Powered by Multiple Linear Regression | Medical Cost Personal Dataset")
