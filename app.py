import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and assets with caching
@st.cache_resource
def load_assets():
    model = joblib.load("student_pass_fail_model.pkl")
    feature_info = joblib.load("feature_info.pkl")
    return model, feature_info

model, feature_info = load_assets()
features = feature_info["features"]

# Custom CSS for Modern SaaS UI
st.markdown("""
<style>
    .stApp { background-color: #0F172A; }
    .gradient-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white; font-weight: 600; border-radius: 10px; height: 3.2em; border: none;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='gradient-title'>🎓 Student Performance Prediction Portal</h1>", unsafe_allow_html=True)

user_input = {}

# User-Friendly Input Form
with st.form("prediction_form"):
    st.markdown("### 📝 Enter Student Information")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Student ID
        student_id_val = st.number_input("🔢 Student ID", min_value=1000, max_value=9999, value=2390, step=1)
        user_input["StudentID"] = int(student_id_val)

        # Age
        user_input["Age"] = st.slider("🎂 Age", min_value=15, max_value=22, value=17)

        # Gender
        gender_choice = st.selectbox("👤 Gender", ["Female", "Male"])
        user_input["Gender"] = 1 if gender_choice == "Male" else 0

        # Ethnicity
        ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
        eth_choice = st.selectbox("🌍 Ethnicity", list(ethnicity_map.keys()))
        user_input["Ethnicity"] = ethnicity_map[eth_choice]

        # Parental Education
        edu_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}
        edu_choice = st.selectbox("🎓 Parental Education Level", list(edu_map.keys()), index=2)
        user_input["ParentalEducation"] = edu_map[edu_choice]

        # Weekly Study Time
        user_input["StudyTimeWeekly"] = st.slider("📚 Weekly Study Time (Hours)", min_value=0.0, max_value=40.0, value=18.0, step=0.5)

        # Absences (Critical Factor)
        user_input["Absences"] = st.number_input("🚨 Absences (School Days Missed)", min_value=0, max_value=30, value=1, help="0 to 3 for best performance")

    with col2:
        # Tutoring
        tut_choice = st.selectbox("👨‍🏫 Receives Tutoring?", ["No", "Yes"], index=1)
        user_input["Tutoring"] = 1 if tut_choice == "Yes" else 0

        # Parental Support
        supp_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
        supp_choice = st.selectbox("🤝 Parental Support Level", list(supp_map.keys()), index=3)
        user_input["ParentalSupport"] = supp_map[supp_choice]

        # Extracurricular
        extra_choice = st.selectbox("🎨 Extracurricular Activities?", ["No", "Yes"], index=1)
        user_input["Extracurricular"] = 1 if extra_choice == "Yes" else 0

        # Sports
        sport_choice = st.selectbox("⚽ Participates in Sports?", ["No", "Yes"], index=1)
        user_input["Sports"] = 1 if sport_choice == "Yes" else 0

        # Music
        music_choice = st.selectbox("🎵 Participates in Music?", ["No", "Yes"], index=1)
        user_input["Music"] = 1 if music_choice == "Yes" else 0

        # Volunteering
        vol_choice = st.selectbox("🙋 Volunteering Work?", ["No", "Yes"], index=1)
        user_input["Volunteering"] = 1 if vol_choice == "Yes" else 0

    st.write("")
    submitted = st.form_submit_button("🔮 Predict Academic Result", use_container_width=True)

# Prediction Result Handler
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        fail_prob = probabilities[0] * 100
        pass_prob = probabilities[1] * 100

        st.divider()
        st.markdown("### 📊 Prediction Dashboard")

        res_col, m1, m2 = st.columns([1.5, 1, 1], gap="medium")

        with res_col:
            if prediction == 1:
                st.success("### 🎉 Prediction: PASS")
                st.caption("The student shows positive academic indicators.")
            else:
                st.error("### ⚠️ Prediction: FAIL")
                st.caption("The student is highlighted as academically at-risk.")

        with m1:
            st.metric("Pass Confidence", f"{pass_prob:.2f}%")
        with m2:
            st.metric("Fail Risk", f"{fail_prob:.2f}%")

        st.write("**Confidence Breakdown:**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Pass Probability: {pass_prob:.1f}%")
            st.progress(int(pass_prob))
        with c2:
            st.caption(f"Fail Probability: {fail_prob:.1f}%")
            st.progress(int(fail_prob))

    except Exception as e:
        st.error("Prediction failed. Please verify input data.")
        st.code(str(e))