#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py

import streamlit as st
import pandas as pd
import joblib
from counterfactual import generate_demurrage_counterfactual

try:
    model = joblib.load('model.pkl')
    df = pd.read_excel('cleaned_data.xlsx')
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.stop()


# Load model and data
model = joblib.load('model.pkl')
df = pd.read_excel('cleaned_data.xlsx')

# Page setup
st.set_page_config(page_title="Demurrage Prediction Tool", layout="centered")
st.title("⛵ Demurrage Prediction and Suggestions")

# Input form
st.subheader("📥 Enter Shipping Details")

quantity = st.number_input("Quantity (MT)")
free_time = st.selectbox("Free Time (Hours)", [6, 24, 27])
discharge_rate = st.number_input("Discharge Rate (MT/hr)")
demurrage_rate = st.number_input("Demurrage Rate Per Day (USD)")

if st.button("🔍 Predict & Suggest"):
    input_data = {
        'Quantity': quantity,
        'Free_Time_Hours': free_time,
        'Discharge_Rate': discharge_rate,
        'Demurrage_Rate_Per_day': demurrage_rate
    }

    with st.spinner("Analyzing..."):
        result = generate_demurrage_counterfactual(input_data, df, model)

    st.subheader("🔎 Results")
    st.write(result)

