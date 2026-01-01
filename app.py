import streamlit as st
import pandas as pd
import pickle

## GETTING INPUTS

st.title("STUDENT DROPOUT RISK PREDICTOR")

st.subheader("STUDENT INFO")

name= st.text_input("STUDENT NAME", placeholder='name')
admission_grade= st.number_input("ADMISSION GRADE", min_value=0, step=1)
debtor= st.selectbox("DEBTOR(Y/N)", options=[0,1], format_func= lambda x: "no" if x==0 else "yes")
tuition_fee= st.selectbox("FEE PENDING(Y/N)", options=[0,1], format_func= lambda x: "no" if x==0 else "yes")
age= st.number_input("ENROLLMENT AGE", min_value=16, step=1)
international= st.selectbox("INTERNATIONAL STUDENT(Y/N)", options=[0,1], format_func= lambda x: "no" if x==0 else "yes")

st.subheader("PARENT INFO")

parent_edu = st.slider("PARENT EDUCATION LEVEL (0-5)", 0, 5)
parent_risk = st.selectbox("PARENT EMPLOYABILITY RISK", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])

st.subheader("ACADEMIC INFO")

course_code= st.number_input("COURSE CODE", min_value=0, step=1)
attendance= st.selectbox("ATTENDANCE TYPE", options=[0,1], format_func= lambda x: "morning" if x==0 else "evening")
scholarship= st.selectbox("SCHLORSHIP HOLDER(Y/N)", options=[0,1], format_func= lambda x: "no" if x==0 else "yes")
approved_credits_1 = st.number_input("APPROVED CREDITS 1ST SEM", min_value=0, step=1)
enrolled_credits_1 = st.number_input("ENROLLED CREDITS 1ST SEM", min_value=0, step=1)

if enrolled_credits_1 == 0:
    approval_rate_1 = 0.0
else:
    approval_rate_1 = approved_credits_1 / enrolled_credits_1

st.write(f"approval rate after first sem: {approval_rate_1}")

approved_credits_2 = st.number_input("APPROVED CREDITS 2ND SEM", min_value=0, step=1)
enrolled_credits_2 = st.number_input("ENROLLED CREDITS 2ND SEM", min_value=0, step=1)

total_enrolled_credits= enrolled_credits_1 + enrolled_credits_2

total_approved_credits= approved_credits_1 + approved_credits_2

if enrolled_credits_2== 0:
    approval_rate_end= 0.0
else:
    approval_rate_end= total_approved_credits/ total_enrolled_credits

st.write(f"approval rate after 2 sems: {approval_rate_end}")
 

## CONSOLIDATE THE INPUT DATA AND PASS INTO OUR PIPELINE

input_data = {
    'Course': course_code,
    'Daytime/evening attendance\t': attendance,
    'Admission grade': admission_grade,
    'Debtor': debtor,
    'Tuition fees up to date': tuition_fee,
    'Scholarship holder': scholarship,
    'Age at enrollment': age,
    'International': international,
    'parent education': parent_edu,
    'parent_employability_risk': parent_risk,
    'approval rate 1st sem': approval_rate_1,
    'approval rate yearend': approval_rate_end
}

with open("model_pipeline.pkl", "rb") as file:
    model= pickle.load(file)

if st.button("Predict Dropout Risk"):
    input_df= pd.DataFrame([input_data])
    dropout_prob = model.predict_proba(input_df)[0][1]

    threshold = 0.3

    st.subheader("📊 Prediction Result")
    st.write(f"**Dropout Probability:** {dropout_prob:.2%}")

    if dropout_prob >= threshold:
        st.error("⚠️ At Risk of Dropout")
    else:
        st.success("✅ Low Risk of Dropout")
    




