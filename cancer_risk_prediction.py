import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.08);
    }
    .result-box-high {
        background-color: #FDECEC;
        border-left: 6px solid #DC2626;
        padding: 20px;
        border-radius: 10px;
    }
    .result-box-low {
        background-color: #E9F9EE;
        border-left: 6px solid #16A34A;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('cancer_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# Known model performance (from evaluation on test set)
MODEL_ACCURACY = 94.60
MODEL_PRECISION = 0.92
MODEL_RECALL = 0.90
MODEL_F1 = 0.91

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🩺 Cancer Risk Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning based risk assessment using Logistic Regression</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL INFO CARDS
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>{MODEL_ACCURACY}%</h3>Accuracy</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>{MODEL_PRECISION}</h3>Precision</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>{MODEL_RECALL}</h3>Recall</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h3>{MODEL_F1}</h3>F1-Score</div>", unsafe_allow_html=True)

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
# PREDICTION
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

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.markdown(f"""
                <div class='result-box-high'>
                <h3>⚠️ High Cancer Risk</h3>
                <p>Estimated probability: <b>{probability*100:.2f}%</b></p>
                <p>This is a model-based estimate, not a medical diagnosis. Please consult a doctor.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box-low'>
                <h3>✅ Low Cancer Risk</h3>
                <p>Estimated probability: <b>{probability*100:.2f}%</b></p>
                <p>This is a model-based estimate, not a medical diagnosis.</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("**Risk Probability**")
        st.progress(float(probability))

    with right:
        st.subheader("Input Summary")
        st.write(f"**Age:** {age}  |  **Gender:** {gender}")
        st.write(f"**BMI:** {bmi}  |  **Genetic Risk:** {['Low','Medium','High'][genetic_risk]}")
        st.write(f"**Smoking:** {smoking}  |  **Alcohol:** {alcohol}")
        st.write(f"**Activity:** {activity} hrs/week  |  **Pollution:** {pollution}/9")
        st.write(f"**Lung Disease:** {'Yes' if lung_disease else 'No'}  |  "
                 f"**Tumor History:** {'Yes' if tumor_history else 'No'}  |  "
                 f"**Family History:** {'Yes' if family_history else 'No'}")

else:
    st.info("👈 Sidebar mein apni details fill karo aur **Predict Cancer Risk** button dabao.")

# ---------------------------------------------------------
# FEATURE IMPORTANCE (Logistic Regression Coefficients)
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📊 Model ke important factors dekhein"):
    feature_names = ['Age', 'Gender', 'BMI', 'Genetic_Risk', 'Physical_Activity_Hours_Per_Week',
                      'Pollution_Exposure_Score', 'Chronic_Lung_Disease', 'Previous_Tumor_History',
                      'Family_History_Cancer', 'Smoking_Status_Former', 'Smoking_Status_Never',
                      'Alcohol_Consumption_Moderate', 'Alcohol_Consumption_None']

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    st.write("Positive coefficient = risk badhata hai, Negative = risk ghataata hai")
    st.bar_chart(coef_df.set_index('Feature'))

st.markdown("---")
st.caption("⚠️ Disclaimer: Ye tool sirf educational/demo purpose ke liye hai, medical advice ka substitute nahi hai.")
