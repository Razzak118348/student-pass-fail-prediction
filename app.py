
# import streamlit as st
# import pandas as pd
# import joblib

# st.set_page_config(
#     page_title="Student Pass/Fail Prediction",
#     page_icon="🎓",
#     layout="centered"
# )

# # Load trained model and feature information
# model = joblib.load("student_pass_fail_model.pkl")
# feature_info = joblib.load("feature_info.pkl")

# features = feature_info["features"]
# numerical_features = feature_info["numerical_features"]
# categorical_features = feature_info["categorical_features"]

# # ---------------------------------------------------------
# # Header
# # ---------------------------------------------------------

# st.title("🎓 Student Pass/Fail Prediction")
# st.write("Enter student information to predict the academic result.")

# st.divider()

# # ---------------------------------------------------------
# # Input Form
# # ---------------------------------------------------------

# user_input = {}

# with st.form("prediction_form"):

#     st.subheader("Student Information")

#     # Numerical features
#     for feature in numerical_features:

#         value = st.text_input(
#             feature,
#             value="0"
#         )

#         try:
#             if feature.lower() == "studentid":
#                 user_input[feature] = int(float(value))
#             else:
#                 user_input[feature] = float(value)

#         except ValueError:
#             st.error(f"Please enter a valid value for {feature}.")
#             user_input[feature] = 0.0

#     # Categorical features
#     for feature in categorical_features:

#         user_input[feature] = st.text_input(
#             feature,
#             value=""
#         )

#     submitted = st.form_submit_button(
#         "🔮 Predict Result",
#         use_container_width=True
#     )

# # ---------------------------------------------------------
# # Prediction
# # ---------------------------------------------------------

# if submitted:

#     try:

#         input_df = pd.DataFrame([user_input])

#         # Same feature order as training
#         input_df = input_df[features]

#         prediction = model.predict(input_df)[0]

#         probabilities = model.predict_proba(input_df)[0]

#         fail_probability = probabilities[0] * 100
#         pass_probability = probabilities[1] * 100

#         st.divider()

#         if prediction == 1:

#             st.success("## ✅ PASS")

#         else:

#             st.error("## ❌ FAIL")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric(
#                 "Pass Probability",
#                 f"{pass_probability:.2f}%"
#             )

#         with col2:
#             st.metric(
#                 "Fail Probability",
#                 f"{fail_probability:.2f}%"
#             )

#     except Exception as e:

#         st.error("Prediction failed.")
#         st.code(str(e))


import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Pass/Fail Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load trained model and feature information with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("student_pass_fail_model.pkl")
    feature_info = joblib.load("feature_info.pkl")
    return model, feature_info

model, feature_info = load_assets()

features = feature_info["features"]
numerical_features = feature_info["numerical_features"]
categorical_features = feature_info["categorical_features"]

# ---------------------------------------------------------
# Custom CSS for Modern UI/UX
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Aesthetic Background */
    .stApp {
        background-color: #0F172A;
    }

    /* Gradient Typography */
    .gradient-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Primary Submit Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 16px;
        border-radius: 12px;
        height: 3.2em;
        border: none;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
    }

    /* Sidebar Clean Cards */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Information Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 System Overview")
    st.caption("AI-driven Predictive Analytics Portal")

    st.markdown("---")
    st.markdown("**Dataset Parameters:**")
    st.markdown(f"• **Total Features:** `{len(features)}`")
    st.markdown(f"• **Numerical Inputs:** `{len(numerical_features)}`")
    st.markdown(f"• **Categorical Inputs:** `{len(categorical_features)}`")

    st.markdown("---")
    st.info("💡 **Tip:** Ensure numerical values are accurately typed for precise model outputs.")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown("<h1 class='gradient-title'>🎓 Student Pass/Fail Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 35px;'>Enter academic details below to predict performance with machine learning models.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Form (Clean 2-Column Grid Layout)
# ---------------------------------------------------------
user_input = {}

with st.form("prediction_form"):
    st.markdown("### 📝 Student Parameters")

    # Grid structure for forms
    col1, col2 = st.columns(2, gap="medium")

    # Process Numerical Features across 2 columns
    for idx, feature in enumerate(numerical_features):
        target_col = col1 if idx % 2 == 0 else col2

        with target_col:
            val = st.text_input(
                label=f"🔢 {feature}",
                value="0",
                key=f"num_{feature}"
            )
            try:
                if feature.lower() == "studentid":
                    user_input[feature] = int(float(val))
                else:
                    user_input[feature] = float(val)
            except ValueError:
                st.error(f"Please enter a valid number for {feature}.")
                user_input[feature] = 0.0

    # Process Categorical Features across 2 columns
    for idx, feature in enumerate(categorical_features):
        target_col = col1 if idx % 2 == 0 else col2

        with target_col:
            user_input[feature] = st.text_input(
                label=f"🏷️ {feature}",
                value="",
                key=f"cat_{feature}"
            )

    st.write("")
    submitted = st.form_submit_button(
        "🔮 Run Prediction Analysis",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction Logic & Result Visualizer
# ---------------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        fail_probability = probabilities[0] * 100
        pass_probability = probabilities[1] * 100

        st.divider()
        st.markdown("### 📈 Analysis Results")

        res_col, m1, m2 = st.columns([1.5, 1, 1], gap="medium")

        with res_col:
            if prediction == 1:
                st.success("### ✅ Prediction: PASS")
                st.caption("The student meets the required benchmarks to pass.")
            else:
                st.error("### ❌ Prediction: FAIL")
                st.caption("The student is highlighted as academically at-risk.")

        with m1:
            st.metric(
                label="Pass Confidence",
                value=f"{pass_probability:.2f}%"
            )

        with m2:
            st.metric(
                label="Fail Risk",
                value=f"{fail_probability:.2f}%"
            )

        # Visual Indicators
        st.markdown("---")
        st.write("**Confidence Distribution:**")

        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Pass Probability: {pass_probability:.1f}%")
            st.progress(int(pass_probability))

        with c2:
            st.caption(f"Fail Probability: {fail_probability:.1f}%")
            st.progress(int(fail_probability))

    except Exception as e:
        st.error("Prediction execution failed. Please check inputs.")
        st.code(str(e))