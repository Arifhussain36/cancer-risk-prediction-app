"""
Cancer Risk Prediction - Multi-Model Interactive App (Professional UI, v2)
Run: streamlit run multi_model_cancer_app.py
Requires: model_logistic.pkl, model_decision_tree.pkl, model_knn.pkl,
          model_xgboost.pkl, scaler.pkl, model_metrics.pkl
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cancer Risk Prediction | AI Multi-Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Force the main content area to a consistent light background, but leave the
       header/toolbar (top-right menu, Deploy button) untouched so its icons stay visible */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }

    /* Streamlit's own menus/popovers (the 3-dot settings menu, dropdown lists) - force readable */
    div[data-baseweb="popover"], div[data-baseweb="popover"] *,
    [role="listbox"], [role="listbox"] * {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    .hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #6D28D9 100%);
        padding: 40px 30px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 8px 24px rgba(30, 58, 138, 0.25);
    }
    .hero h1 { color: white !important; font-size: 2.6rem; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.5px; }
    .hero p { color: #E0E7FF !important; font-size: 1.05rem; }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 10px;
        margin-bottom: 14px;
        border-left: 5px solid #6D28D9;
        padding-left: 12px;
    }

    /* Big model selector cards (via buttons) - plain white always, no hover color change.
       Using data-testid (stable across Streamlit versions) instead of the old .stButton class */
    div[data-testid="column"] [data-testid="stButton"] button,
    div[data-testid="column"] [data-testid="stButton"] button:hover,
    div[data-testid="column"] [data-testid="stButton"] button:focus,
    div[data-testid="column"] [data-testid="stButton"] button p {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        font-weight: 700 !important;
        border-radius: 16px !important;
        padding: 22px 10px !important;
        border: 2px solid #E5E7EB !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    }

    .result-box-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 8px solid #DC2626;
        padding: 25px;
        border-radius: 16px;
    }
    .result-box-low {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border-left: 8px solid #16A34A;
        padding: 25px;
        border-radius: 16px;
    }
    .result-box-high h2, .result-box-low h2 { font-weight: 800; font-size: 1.6rem; margin-bottom: 6px; }
    .result-box-high h2 { color: #991B1B !important; }
    .result-box-low h2 { color: #166534 !important; }
    .result-box-high p, .result-box-low p { color: #1F2937 !important; }

    /* Sidebar: force readable colors in both light and dark mode */
    [data-testid="stSidebar"] {
        background-color: #F5F3FF !important;
    }
    [data-testid="stSidebar"] * {
        color: #1F2937 !important;
    }
    /* Force light background on actual input widgets so dark text stays visible in dark mode */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    /* Predict button - full width, bold, purple gradient, white text, normal height */
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] [data-testid="stButton"] button p {
        color: white !important;
        width: 100% !important;
        background: linear-gradient(135deg, #6D28D9, #1E3A8A) !important;
        border: none !important;
        font-size: 1.05rem !important;
        padding: 10px 0px !important;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load('model_logistic.pkl'),
        "Decision Tree": joblib.load('model_decision_tree.pkl'),
        "KNN": joblib.load('model_knn.pkl'),
        "XGBoost": joblib.load('model_xgboost.pkl'),
    }
    scaler = joblib.load('scaler.pkl')
    metrics = joblib.load('model_metrics.pkl')
    return models, scaler, metrics

models, scaler, metrics = load_artifacts()

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown("""
    <div class="hero">
        <h1>🧬 Cancer Risk Prediction</h1>
        <p>AI-powered multi-model risk assessment — compare 4 machine learning algorithms in real time</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL SELECTOR (big clickable cards)
# ---------------------------------------------------------
st.markdown("<div class='section-title'>Select a Model</div>", unsafe_allow_html=True)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Logistic Regression"

model_icons = {"Logistic Regression": "📈", "Decision Tree": "🌳", "KNN": "🔵", "XGBoost": "⚡"}

cols = st.columns(4)
for i, (name, m) in enumerate(metrics.items()):
    with cols[i]:
        label = f"{model_icons.get(name,'')}  {name}"
        if st.button(label, key=f"btn_{name}"):
            st.session_state.selected_model = name

selected_model_name = st.session_state.selected_model

# Dynamically highlight the selected model's card with a grey/purple shade
selected_index = list(models.keys()).index(selected_model_name) + 1  # nth-of-type is 1-based
st.markdown(f"""
    <style>
    div[data-testid="column"]:nth-of-type({selected_index}) [data-testid="stButton"] button,
    div[data-testid="column"]:nth-of-type({selected_index}) [data-testid="stButton"] button p {{
        background-color: #E5E7EB !important;
        border: 2px solid #6D28D9 !important;
        color: #6D28D9 !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.success(f"✅ Currently selected: **{selected_model_name}**")

# ---------------------------------------------------------
# DONUT GAUGES: Accuracy, Precision, Recall, F1 for SELECTED model
# ---------------------------------------------------------
sel = metrics[selected_model_name]
st.markdown(f"<div class='section-title'>📈 {selected_model_name} — Detailed Performance</div>", unsafe_allow_html=True)

gauge_data = [
    ("Accuracy", sel['accuracy'] * 100, "#6D28D9"),
    ("Precision", sel['precision'] * 100, "#2563EB"),
    ("Recall", sel['recall'] * 100, "#0D9488"),
    ("F1-Score", sel['f1'] * 100, "#EA580C"),
]

gcols = st.columns(4)
for i, (label, value, color) in enumerate(gauge_data):
    with gcols[i]:
        fig = go.Figure(data=[go.Pie(
            values=[value, 100 - value],
            hole=0.7,
            marker=dict(colors=[color, "#F1F5F9"]),
            textinfo="none",
            sort=False,
            direction="clockwise"
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=160,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f"{value:.1f}%", x=0.5, y=0.55, font_size=18,
                               font_family="Poppins", font_color=color, showarrow=False),
                         dict(text=label, x=0.5, y=0.30, font_size=12,
                               font_family="Poppins", font_color="#555", showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("📋 Patient Information")

age = st.sidebar.slider("Age", 18, 90, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)
smoking = st.sidebar.selectbox("Smoking Status", ["Current", "Former", "Never"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["Heavy", "Moderate", "None"])
genetic_risk = st.sidebar.select_slider("Genetic Risk", options=[0, 1, 2],
                                         format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
activity = st.sidebar.slider("Physical Activity (hrs/week)", 0.0, 35.0, 5.0)
pollution = st.sidebar.slider("Pollution Exposure Score", 1, 9, 5)

st.sidebar.markdown("**Medical History**")
lung_disease = st.sidebar.checkbox("Chronic Lung Disease")
tumor_history = st.sidebar.checkbox("Previous Tumor History")
family_history = st.sidebar.checkbox("Family History of Cancer")

predict_btn = st.sidebar.button("🔍 Predict Cancer Risk", use_container_width=True)

# ---------------------------------------------------------
# PREDICTION + DONUT CHART
# ---------------------------------------------------------
if predict_btn:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "Female" else 0],
        'BMI': [bmi],
        'Genetic_Risk': [genetic_risk],
        'Physical_Activity_Hours_Per_Week': [activity],
        'Pollution_Exposure_Score': [pollution],
        'Chronic_Lung_Disease': [int(lung_disease)],
        'Previous_Tumor_History': [int(tumor_history)],
        'Family_History_Cancer': [int(family_history)],
        'Smoking_Status_Former': [1 if smoking == "Former" else 0],
        'Smoking_Status_Never': [1 if smoking == "Never" else 0],
        'Alcohol_Consumption_Moderate': [1 if alcohol == "Moderate" else 0],
        'Alcohol_Consumption_None': [1 if alcohol == "None" else 0],
    })

    cols_to_scale = ['Age', 'BMI', 'Genetic_Risk', 'Physical_Activity_Hours_Per_Week', 'Pollution_Exposure_Score']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    selected_model = models[selected_model_name]
    prediction = selected_model.predict(input_data)[0]
    probability = selected_model.predict_proba(input_data)[0][1]
    risk_pct = probability * 100
    safe_pct = 100 - risk_pct

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
                <div class='result-box-high'>
                <h2>⚠️ High Cancer Risk</h2>
                <p>Model used: <b>{selected_model_name}</b> (Accuracy: {sel['accuracy']*100:.2f}%)</p>
                <p>Estimated risk probability: <b>{risk_pct:.2f}%</b></p>
                <p>This is a model-based estimate, not a medical diagnosis. Please consult a doctor.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box-low'>
                <h2>✅ Low Cancer Risk</h2>
                <p>Model used: <b>{selected_model_name}</b> (Accuracy: {sel['accuracy']*100:.2f}%)</p>
                <p>Estimated risk probability: <b>{risk_pct:.2f}%</b></p>
                <p>This is a model-based estimate, not a medical diagnosis.</p>
                </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-title'>Risk Probability</div>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=["Risk", "Safe"],
            values=[risk_pct, safe_pct],
            hole=0.65,
            marker=dict(colors=["#DC2626", "#16A34A"]),
            textinfo="label+percent",
            textfont=dict(size=14, family="Poppins"),
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f"{risk_pct:.1f}%", x=0.5, y=0.5,
                               font_size=28, font_family="Poppins", font_color="#1E3A8A",
                               showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Fill in the patient details in the sidebar and click the **Predict** button.")

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational/demo purposes only and is not a substitute for medical advice.")
