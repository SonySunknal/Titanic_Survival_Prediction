import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

@st.cache_resource
def load_model(path="models/titanic_pipe.joblib"):
    return joblib.load(path)

model = load_model()
st.title("Titanic Survival Predictor")

pclass = st.selectbox("Pclass (1=Upper, 2=Mid, 3=Lower)", [1,2,3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch =st.number_input("Parents/Children abroad (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr","Mrs", "Miss", "Master", "Rare", "Unknown"])
family_size = st.number_input("FamilySize", min_value=1, max_value=20, value=1)
is_alone = 1 if family_size==1 else 0
has_cabin = st.selectbox("Has Cabin?", [0,1])

input_df = pd.DataFrame([{"Pclass": pclass, "Sex":sex, "Age":age, "Sibsp":sibsp, "Fare":fare, "Embarked":embarked, "Title": title, "FamilySize": family_size, "IsAlone":is_alone, "HasCabin":has_cabin}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.write("**Prediction:**", "Survived" if pred==1 else "Did not survive")
    st.write(f"**Survival probability:** {proba:.2f}")
    