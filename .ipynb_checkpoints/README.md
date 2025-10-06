# 🚢 Titanic Survival Prediction

A Machine Learning project that predicts whether a passenger survived the Titanic disaster based on personal and travel details.

---

## 📘 Project Overview
This project uses the **Titanic dataset** from Kaggle.  
We trained a model using **scikit-learn**, built a **Streamlit web app**, and deployed it using **Streamlit Community Cloud**.

---

## 🧠 Tech Stack
- **Python 3.10+**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Joblib**
- **Streamlit**

---

## ⚙️ How It Works
1. **Data Preprocessing** – handled missing values and categorical features  
2. **Feature Engineering** – added new features like Title, FamilySize, IsAlone, etc.  
3. **Model Training** – used a `RandomForestClassifier` with pipeline  
4. **Deployment** – built an interactive Streamlit app

---

## 📁 Project Structure
app.py
models/titanic_pipe.joblib
requirements.txt
.gitignore
README.md