from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page Configuration with exact project name and medical/particle emojis
st.set_page_config(
    page_title="Medical Insurance Cost Prediction", 
    page_icon="🧬", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Attractive Custom CSS with vibrant gradient and particle-like styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 700;
        font-size: 16px;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    h1 {
        color: #1e1b4b;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h3 {
        color: #312e81;
        font-size: 1.2rem !important;
        font-weight: 600;
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
    elif 18.5 <= b < 25:
        return "Normal & Healthy 🟢"
    elif 25 <= b < 30:
        return "Overweight 🟠"
    else:
        return "Obese 🔴"

if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75 = train_model()

# ---------------- Header Section ----------------
st.title("🧬 Medical Insurance Cost Prediction ✨")
st.markdown("💊 *Enter the customer details below to get an accurate estimation of yearly insurance expenses using Machine Learning.* 🩺")
st.divider()

# Input Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Patient Details")
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    # Default BMI set to 22.0 (Normal) so it doesn't show overweight by default
    bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=22.0, step=0.1)
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0)

with col2:
    st.markdown("### 📋 Lifestyle & Region")
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.info(f"✨ **BMI Status:** {bmi_category(bmi)}")
st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Predict Insurance Cost", type="primary", use_container_width=True):
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
        level = "Low Cost Tier 🟢"
    elif cost < q75:
        level = "Medium Cost Tier 🟡"
    else:
        level = "High Cost Tier 🔴"

    st.divider()
    st.subheader("🎯 Prediction Result")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Estimated Cost (USD)", value=f"${cost:,.2f}")
    with res_col2:
        st.metric(label="Cost Category", value=level)

    if cost < q50:
        st.success(f"Estimated Insurance Cost: **${cost:,.2f}** (Budget Friendly Tier)")
    elif cost < q75:
        st.warning(f"Estimated Insurance Cost: **${cost:,.2f}** (Moderate Tier)")
    else:
        st.error(f"Estimated Insurance Cost: **${cost:,.2f}** (High Risk Tier)")

    with st.expander("🔍 View Raw Model Input"):
        st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("✨ Medical Insurance Cost Prediction App • Built with Streamlit & Machine Learning 🚀")
