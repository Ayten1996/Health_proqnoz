import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
le_group = joblib.load('le_group.pkl')
le_cat = joblib.load('le_cat.pkl')

st.title("Risk Proqnozlaşdırma App")


fast_blood_sugar = st.number_input('fasting blood sugar', value=97)
cholesterol = st.number_input('Cholesterol', value=239)
ldl = st.number_input('LDL', value=142)
hdl = st.number_input('HDL', value=70)
ast = st.number_input('AST', value=61)
alt = st.number_input('ALT', value=115)
serum_creatinine = st.number_input('serum creatinine', value=1.0)

if st.button('Proqnoz et'):
    data = pd.DataFrame([{
        'fast_blood_sugar': fast_blood_sugar,
        'cholesterol': cholesterol,
        'ldl': ldl,
        'hdl': hdl,
        'ast': ast,
        'alt': alt,
        'serum_creatinine': serum_creatinine
    }])
    preds = model.predict(data)
    risk_group = le_group.inverse_transform([preds[0][0]])[0]
    risk_cat = le_cat.inverse_transform([preds[1][0]])[0]
    st.success(f"Risk Group: {risk_group}")
    st.success(f"Risk Category: {risk_cat}")

   
