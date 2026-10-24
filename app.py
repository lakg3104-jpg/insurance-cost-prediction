from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="centered")

# Corrected path pointing directly to the root directory
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
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"


if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75 = train_model()

# ---------------- Interface ----------------
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter the customer details below to get the estimated yearly insurance cost.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of children", min_value=0, max_value=5, value=0)
with col2:
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.caption(f"BMI category: **{bmi_category(bmi)}**")

if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
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
        level = "Low"
    elif cost < q75:
        level = "Medium"
    else:
        level = "High"

    st.divider()
    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Estimated Insurance Cost", f"${cost:,.2f}")
    c2.metric("Cost level", level)

    if level == "Low":
        st.success(f"Estimated Insurance Cost: ${cost:,.2f}")
    elif level == "Medium":
        st.warning(f"Estimated Insurance Cost: ${cost:,.2f}")
    else:
        st.error(f"Estimated Insurance Cost: ${cost:,.2f}")

    with st.expander("Show input sent to the model"):
        st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("Model: Multiple Linear Regression trained on the Medical Cost Personal Dataset.")
