from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- Page config
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------ CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #0b2b3a;
    --teal: #0e7c86;
    --teal-dark: #0a5f67;
    --mint: #e6f2f1;
    --paper: #f6faf9;
    --line: #d5e4e2;
    --muted: #5b7480;
    --coral: #e4572e;
    --amber: #e9a23b;
    --green: #2f9e6f;
}

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(900px 400px at 85% -10%, #d4ebe8 0%, transparent 60%),
        radial-gradient(700px 400px at -10% 10%, #e9f3f1 0%, transparent 60%),
        var(--paper);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 820px;}

/* ---------- Hero ---------- */
.hero {text-align: center; padding: 8px 0 22px 0;}
.hero .badge {
    display: inline-block;
    background: var(--mint);
    color: var(--teal-dark);
    border: 1px solid var(--line);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.12;
    color: var(--ink);
    margin: 0 0 10px 0;
    padding: 0;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.55;
}

/* ---------- Cards (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"],
div[class*="st-key-card_"] {
    background: #ffffff;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    box-shadow: 0 1px 2px rgba(11, 43, 58, 0.04), 0 12px 28px -12px rgba(11, 43, 58, 0.14);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {padding: 6px 10px;}

.card-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.15rem;
    color: var(--ink);
    margin: 4px 0 2px 0;
}
.card-sub {color: var(--muted); font-size: 0.9rem; margin-bottom: 6px;}

/* ---------- Inputs ---------- */
label, .stSlider label, .stRadio label {font-weight: 600 !important; color: var(--ink) !important;}

div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--teal) !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 0 0 2px var(--teal), 0 4px 10px rgba(14, 124, 134, 0.35) !important;
}
div[data-testid="stSliderThumbValue"], div[data-testid="stSliderThumbValue"] span {
    color: var(--teal-dark) !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: var(--line) !important;
    background: var(--paper) !important;
}
div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {font-weight: 500;}

/* ---------- BMI chip ---------- */
.bmi-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 20px;
    margin: 18px 0 8px 0;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: #ffffff;
}
.bmi-chip .left {display: flex; align-items: center; gap: 12px;}
.bmi-dot {width: 12px; height: 12px; border-radius: 50%;}
.bmi-chip .label {color: var(--muted); font-size: 0.9rem;}
.bmi-chip .value {font-weight: 700; font-size: 1.05rem;}
.bmi-chip .num {font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.3rem; color: var(--ink);}

/* ---------- Primary button ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    border-radius: 14px;
    padding: 0.85rem 1.5rem;
    border: none;
    width: 100%;
    box-shadow: 0 10px 22px -8px rgba(14, 124, 134, 0.55);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 14px 26px -8px rgba(14, 124, 134, 0.65);
}
.stButton > button:focus-visible {outline: 3px solid #9fd6d2; outline-offset: 2px;}

/* ---------- Result panel ---------- */
.result {
    background: linear-gradient(160deg, var(--ink) 0%, #12435a 100%);
    color: #ffffff;
    border-radius: 22px;
    padding: 30px 32px;
    margin-top: 10px;
    box-shadow: 0 24px 48px -20px rgba(11, 43, 58, 0.55);
}
.result .kicker {color: #9cc9cc; font-size: 0.95rem; font-weight: 500;}
.result .amount {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 3.2rem;
    line-height: 1.1;
    margin: 4px 0 14px 0;
}
.tier {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    color: #ffffff;
}

/* Gauge */
.gauge {position: relative; height: 12px; border-radius: 999px; margin: 28px 0 8px 0;
        background: linear-gradient(90deg, var(--green) 0%, var(--amber) 55%, var(--coral) 100%);}
.gauge .pin {
    position: absolute; top: -6px; width: 24px; height: 24px; border-radius: 50%;
    background: #ffffff; border: 4px solid var(--ink); transform: translateX(-50%);
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.gauge-scale {display: flex; justify-content: space-between; color: #9cc9cc; font-size: 0.82rem;}

/* Small stat tiles */
.stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 22px;}
.stat {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 14px;
    padding: 14px 16px;
}
.stat .s-label {color: #9cc9cc; font-size: 0.8rem;}
.stat .s-value {font-weight: 700; font-size: 1.15rem; margin-top: 2px;}

@media (max-width: 640px) {
    .hero h1 {font-size: 2rem;}
    .result .amount {font-size: 2.4rem;}
    .stats {grid-template-columns: 1fr;}
}
@media (prefers-reduced-motion: reduce) {
    .stButton > button {transition: none;}
}

.footer {text-align: center; color: #8aa0a8; font-size: 0.88rem; margin-top: 34px;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- Model
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
    return model, X.columns, y.quantile(0.50), y.quantile(0.75), y.mean(), y.max()


def bmi_category(b):
    if b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Normal"
    elif b < 30:
        return "Overweight"
    return "Obese"


BMI_COLORS = {
    "Underweight": "#2b8fd6",
    "Normal": "#2f9e6f",
    "Overweight": "#e9a23b",
    "Obese": "#e4572e",
}

def flat(html: str) -> str:
    """Remove indentation and blank lines so Markdown never treats HTML as a code block."""
    return "".join(line.strip() for line in html.splitlines())


if not DATA_FILE.exists():
    st.error("insurance_cleaned.csv not found in the root directory.")
    st.stop()

model, columns, q50, q75, avg_cost, max_cost = train_model()

# ----------------------------------------------------------------------- Hero
st.markdown(
    """
<div class="hero">
    <div class="badge">🏥 Healthcare cost estimator</div>
    <h1>Medical Insurance Cost Prediction</h1>
    <p>Enter a few patient details and get an estimated yearly insurance cost, based on a model trained on real insurance records.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- Form
col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True, key="card_body"):
        st.markdown(
            '<div class="card-title">Body details</div><div class="card-sub">Age, weight and family size</div>',
            unsafe_allow_html=True,
        )
        age = st.slider("Patient Age", min_value=18, max_value=64, value=30)
        bmi = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=22.0, step=0.1)
        children = st.slider("Number of Dependents", min_value=0, max_value=5, value=0)

with col2:
    with st.container(border=True, key="card_lifestyle"):
        st.markdown(
            '<div class="card-title">Lifestyle and location</div><div class="card-sub">Habits and where the patient lives</div>',
            unsafe_allow_html=True,
        )
        sex = st.radio("Gender", ["female", "male"], horizontal=True)
        smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
        region = st.selectbox("Residential Region", ["northeast", "northwest", "southeast", "southwest"])

# ------------------------------------------------------------------ BMI chip
status = bmi_category(bmi)
color = BMI_COLORS[status]
st.markdown(
    flat(f"""
<div class="bmi-chip" style="border-left: 6px solid {color};">
    <div class="left">
        <div class="bmi-dot" style="background:{color};"></div>
        <div>
            <div class="label">Calculated BMI status</div>
            <div class="value" style="color:{color};">{status}</div>
        </div>
    </div>
    <div class="num">{bmi:.1f}</div>
</div>
"""),
    unsafe_allow_html=True,
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Prediction
if st.button("📊 Calculate Insurance Cost", type="primary"):
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "sex": 1 if sex == "male" else 0,
                "bmi": bmi,
                "children": children,
                "smoker": 1 if smoker == "yes" else 0,
                "region_northwest": int(region == "northwest"),
                "region_southeast": int(region == "southeast"),
                "region_southwest": int(region == "southwest"),
            }
        ]
    )[columns]

    cost = max(model.predict(input_data)[0], 0)

    if cost < q50:
        level, tier_color, bracket = "Low cost tier", "#2f9e6f", "Low risk bracket"
    elif cost < q75:
        level, tier_color, bracket = "Medium cost tier", "#e9a23b", "Moderate risk bracket"
    else:
        level, tier_color, bracket = "High cost tier", "#e4572e", "High risk bracket"

    # Gauge position (0-100%) against the dataset's cost range
    pin_pos = min(max(cost / max_cost, 0), 1) * 100
    diff = cost - avg_cost
    diff_txt = f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f}"

    st.markdown(
        flat(f"""
<div class="result">
    <div class="kicker">Estimated annual cost (USD)</div>
    <div class="amount">${cost:,.2f}</div>
    <span class="tier" style="background:{tier_color};">{level}</span>

    <div class="gauge"><div class="pin" style="left:{pin_pos:.1f}%;"></div></div>
    <div class="gauge-scale"><span>$0</span><span>${max_cost:,.0f}</span></div>

    <div class="stats">
        <div class="stat"><div class="s-label">Risk bracket</div><div class="s-value">{bracket}</div></div>
        <div class="stat"><div class="s-label">Average in dataset</div><div class="s-value">${avg_cost:,.0f}</div></div>
        <div class="stat"><div class="s-label">Compared with average</div><div class="s-value">{diff_txt}</div></div>
    </div>
</div>
"""),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("🔍 View raw model feature vector"):
        st.dataframe(input_data, hide_index=True)

# --------------------------------------------------------------------- Footer
st.markdown(
    '<div class="footer">Medical Insurance Cost Prediction • Built with Python and Streamlit</div>',
    unsafe_allow_html=True,
)
