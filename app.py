from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction", 
    page_icon="🏥", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Professional "Wow-Factor" CSS with Card Borders, Shadows, and Sleek Typography
st.markdown("""
    <style>
    .main {
        background-color: #f1f5f9;
    }
    /* Sleek Card Container with Borders */
    .css-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    /* Professional Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.35);
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h3 {
        color: #1e293b;
        font-size: 1.2rem !important;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 16px;
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
        return "Underweight"
    elif 18.5 <= b < 25:
        return "Normal"
    elif 25 <= b < 30:
        return "Overweight"
    else:
        return "Obese"

if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75 = train_model()

# ---------------- Header Section ----------------
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("💡 *An advanced machine learning analytics platform designed to estimate healthcare expenditure securely and accurately.*")
st.divider()

# Input Form in a Professional Card Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Patient Demographics")
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=22.0, step=0.1)
    children = st.slider("Number of Dependents", min_value=0, max_value=5, value=0)

with col2:
    st.markdown("### 📋 Lifestyle & Region")
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
    region = st.selectbox("Residential Region", ["northeast", "northwest", "southeast", "southwest"])

st.info(f"🛡️ **Calculated BMI Status:** {bmi_category(bmi)}")
st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("📊 Calculate Insurance Cost", type="primary"):
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
        level = "Low Cost Tier"
    elif cost < q75:
        level = "Medium Cost Tier"
    else:
        level = "High Cost Tier"

    st.divider()
    st.subheader("📈 Prediction Analysis Report")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Estimated Annual Cost (USD)", value=f"${cost:,.2f}")
    with res_col2:
        st.metric(label="Assigned Risk Tier", value=level)

    if cost < q50:
        st.success(f"Projected annual medical insurance expense: **${cost:,.2f}** (Low Risk Bracket)")
    elif cost < q75:
        st.warning(f"Projected annual medical insurance expense: **${cost:,.2f}** (Moderate Risk Bracket)")
    else:
        st.error(f"Projected annual medical insurance expense: **${cost:,.2f}** (High Risk Bracket)")

    with st.expander("🔍 View Raw Model Feature Vector"):
        st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("✨ Medical Insurance Cost Prediction Platform • Powered by Python & Streamlit 🚀")
