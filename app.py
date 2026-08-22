
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained model and feature information
model = joblib.load("student_pass_fail_model.pkl")
feature_info = joblib.load("feature_info.pkl")

features = feature_info["features"]
numerical_features = feature_info["numerical_features"]
categorical_features = feature_info["categorical_features"]

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🎓 Student Pass/Fail Prediction")
st.write("Enter student information to predict the academic result.")

st.divider()

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------

user_input = {}

with st.form("prediction_form"):

    st.subheader("Student Information")

    # Numerical features
    for feature in numerical_features:

        value = st.text_input(
            feature,
            value="0"
        )

        try:
            if feature.lower() == "studentid":
                user_input[feature] = int(float(value))
            else:
                user_input[feature] = float(value)

        except ValueError:
            st.error(f"Please enter a valid value for {feature}.")
            user_input[feature] = 0.0

    # Categorical features
    for feature in categorical_features:

        user_input[feature] = st.text_input(
            feature,
            value=""
        )

    submitted = st.form_submit_button(
        "🔮 Predict Result",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    try:

        input_df = pd.DataFrame([user_input])

        # Same feature order as training
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        fail_probability = probabilities[0] * 100
        pass_probability = probabilities[1] * 100

        st.divider()

        if prediction == 1:

            st.success("## ✅ PASS")

        else:

            st.error("## ❌ FAIL")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pass Probability",
                f"{pass_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Fail Probability",
                f"{fail_probability:.2f}%"
            )

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))
