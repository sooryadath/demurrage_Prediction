#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py

import streamlit as st
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice
#from counterfactual import generate_demurrage_counterfactual


def generate_demurrage_counterfactual(input_dict, df, model):
   
    feature_cols = [
        'Quantity', 'Free_Time_Hours', 'Discharge_Rate',
        'Demurrage_Rate_Per_day'
    ]
    target_col = 'Demurrage_Incurred'

    # Prepare data for DiCE
    df_dice = df[feature_cols + [target_col]]
    data_dice = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=feature_cols,
        outcome_name=target_col
    )
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(data_dice, model_dice)

    # Convert input_dict to DataFrame
    query_instance = pd.DataFrame([input_dict])

    # Predict class
    pred = model.predict(query_instance)[0]

    if pred == 1:
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0,
                features_to_vary=['Quantity', 'Discharge_Rate']
            )
            return dice_exp.visualize_as_dataframe()
        except Exception as e:
            return pd.DataFrame({'Error': [f"Failed to generate counterfactual: {e}"]})
    else:
        return pd.DataFrame({'Message': ['No demurrage predicted. No counterfactual needed.']})


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
    st.dataframe(result)

