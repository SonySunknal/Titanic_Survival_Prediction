# app.py
import streamlit as st
import pandas as pd
import joblib
import cloudpickle
import os
import traceback

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

@st.cache_resource
def load_model():
    """
    Load model: prefer cloudpickle .pkl (more portable) then fallback to joblib .joblib.
    """
    model_pkl = os.path.join("models", "titanic_pipe.pkl")
    model_joblib = os.path.join("models", "titanic_pipe.joblib")

    if os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            return cloudpickle.load(f)
    elif os.path.exists(model_joblib):
        return joblib.load(model_joblib)
    else:
        raise FileNotFoundError(
            "No model file found. Expected models/titanic_pipe.pkl or models/titanic_pipe.joblib"
        )

# Try loading the model and show a clean error in the UI if it fails.
try:
    model = load_model()
except Exception as e:
    st.error("Model load failed — open Manage app → Logs on Streamlit Cloud for full traceback.")
    st.write("Short error:", str(e))
    print("=== MODEL LOAD TRACEBACK ===")
    traceback.print_exc()
    st.stop()

st.title("Titanic Survival Predictor")

# USER INPUTS — names and capitalization must match training data exactly
pclass = st.selectbox("Pclass (1 = Upper, 2 = Mid, 3 = Lower)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare", "Unknown"])
family_size = st.number_input("FamilySize", min_value=1, max_value=20, value=1)
is_alone = 1 if family_size == 1 else 0
has_cabin = st.selectbox("Has Cabin?", [0, 1])

# Build input dataframe — EXACT column names used when training.
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Title": title,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "HasCabin": has_cabin
}])

st.write("Input preview:")
st.dataframe(input_df)

if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.write("**Prediction:**", "Survived" if pred == 1 else "Did not survive")
        st.write(f"**Survival probability:** {proba:.2f}")
    except Exception as e:
        st.error("Prediction failed — check logs for details.")
        print("=== PREDICTION TRACEBACK ===")
        traceback.print_exc()
        st.stop()

    