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

# Professional "Wow-Factor" CSS with Centering, Clean Card Borders, and Shadows
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    /* Sleek Card Container with Borders */
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
        margin-bottom: 16px;
    }
    /* Professional Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 12px;
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

# ---------------- Centered Header Section ----------------
st.markdown("""
    <div style="text-align: center; padding-top: 10px; padding-bottom: 10px;">
        <h1 style="color: #0f172a; font-weight: 800; font-size: 2.3rem; margin-bottom: 6px;">🏥 Medical Insurance Cost Prediction</h1>
        <p style="color: #64748b; font-size: 1.05rem; font-weight: 400; margin-top: 0px;">
            An advanced machine learning analytics platform designed to estimate healthcare expenditure securely and accurately.
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Input Form in Clean Framed Cards (No awkward headings)
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.slider("Patient Age", min_value=18, max_value=64, value=30)
    bmi = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=22.0, step=0.1)
    children = st.slider("Number of Dependents", min_value=0, max_value=5, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
    region = st.selectbox("Residential Region", ["northeast", "northwest", "southeast", "southwest"])
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Centered Results Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 15px;">
            <h3 style="color: #0f172a; font-weight: 700;">📈 Prediction Analysis Report</h3>
        </div>
    """, unsafe_allow_html=True)
    
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
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
        ✨ Medical Insurance Cost Prediction Platform • Powered by Python & Streamlit 🚀
    </div>
""", unsafe_allow_html=True)
