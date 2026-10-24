from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="MediCare Cost Intelligence Hub", 
    page_icon="🩺", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling (Clean Teal & Healthcare Blue)
st.markdown("""
    <style>
    .main {
        background-color: #f4fbfb;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        border: none;
        box-shadow: 0 4px 10px rgba(13, 148, 136, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0f766e 100%, #115e59 100%);
        box-shadow: 0 6px 15px rgba(13, 148, 136, 0.4);
    }
    h1 {
        color: #134e4a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h3 {
        color: #0f766e;
        font-size: 1.2rem !important;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 10px;
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
        return "Underweight (Need Nutrition Guidance) 🔵"
    if b < 25:
        return "Normal & Healthy 🟢"
    if b < 30:
        return "Overweight 🟠"
    return "Obese (High Risk Tier) 🔴"

if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75 = train_model()

# ---------------- Header Section ----------------
st.title("🩺 MediCare Cost Intelligence Portal")
st.markdown("Welcome to the advanced clinical insurance analytics platform. Enter patient parameters below to generate real-time policy cost projections.")
st.divider()

# Input Form Layout in Clean Cards/Columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Patient Clinical Profile")
    age = st.slider("Patient Age", min_value=18, max_value=64, value=30)
    bmi = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Dependent Children", min_value=0, max_value=5, value=0)

with col2:
    st.markdown("### 📋 Risk & Demographic Factors")
    sex = st.radio("Biological Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Tobacco Smoking Status", ["no", "yes"], horizontal=True)
    region = st.selectbox("Geographic Zone", ["northeast", "northwest", "southeast", "southwest"])

st.info(f"**Evaluated BMI Status:** {bmi_category(bmi)}")
st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action Button
if st.button("Generate Insurance Cost Estimation", type="primary", use_container_width=True):
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
        level = "Standard Tier (Low Risk)"
    elif cost < q75:
        level = "Moderate Tier (Standard Risk)"
    else:
        level = "Premium Tier (High Risk)"

    st.divider()
    st.subheader("📊 Analytical Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Estimated Annual Premium", value=f"${cost:,.2f}")
    with res_col2:
        st.metric(label="Calculated Risk Band", value=level)

    if cost < q50:
        st.success(f"The projected annual insurance expenditure is **${cost:,.2f}** (Optimized Tier).")
    elif cost < q75:
        st.warning(f"The projected annual insurance expenditure is **${cost:,.2f}** (Moderate Risk Tier).")
    else:
        st.error(f"The projected annual insurance expenditure is **${cost:,.2f}** (High Risk Premium Tier).")

    with st.expander("🔍 Inspect Raw Model Vector Input"):
        st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("MediCare Analytics Engine • Powered by Multiple Linear Regression & Streamlit")
