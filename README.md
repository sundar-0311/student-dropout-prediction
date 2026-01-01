# 🎓 Student Dropout Prediction

An end-to-end machine learning project to predict end-of-year student dropout risk using academic and socio-economic factors.  
The project includes a fully trained ML pipeline and a Streamlit web app for real-time predictions.

---

## 🚀 Features
- Feature engineering (semester-wise & year-end approval rates)
- Logistic Regression with preprocessing pipeline
- Threshold tuning to improve dropout recall (71% → 80%)
- Interactive Streamlit app for real-time inference

---

## 🧠 Machine Learning Pipeline
- Data cleaning and feature selection
- Safe approval-rate computation (no division-by-zero errors)
- ColumnTransformer for numerical & categorical preprocessing
- Logistic Regression classifier
- Recall-focused evaluation and threshold tuning

---

## 🖥️ Streamlit App
Users can input student details such as:
- Course & attendance type
- Academic performance (approved/enrolled credits)
- Financial indicators (debtor, fees status)
- Parental background indicators

The app outputs:
- Dropout probability
---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
