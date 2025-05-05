# app.py

import streamlit as st
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice
st.set_page_config(page_title="Demurrage Prediction Tool", layout="centered")
# ------------------------
# Load model and data
# ------------------------
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_data
def load_data():
    return pd.read_excel('cleaned_data.xlsx')

model = load_model()
df = load_data()

# ------------------------
# Generate counterfactual
# ------------------------
def generate_demurrage_counterfactual(input_dict, df, model):
    try:
        feature_cols = ['Quantity', 'Free_Time_Hours', 'Discharge_Rate', 'Demurrage_Rate_Per_day']
        target_col = 'Demurrage_Incurred'

        df_dice = df[feature_cols + [target_col]]
        data_dice = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=feature_cols,
            outcome_name=target_col
        )
        model_dice = dice_ml.Model(model=model, backend="sklearn")
        exp = Dice(data_dice, model_dice)

        query_instance = pd.DataFrame([input_dict])
        pred = model.predict(query_instance)
        if pred is None or len(pred) == 0:
            return pd.DataFrame({'Error': ['Prediction failed.']})
        pred = pred[0]

        if pred == 1:
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0,
                features_to_vary=['Quantity', 'Discharge_Rate']
            )
            df_cf = dice_exp.visualize_as_dataframe()
            return df_cf
        else:
            return pd.DataFrame({'Message': ['✅ No demurrage predicted. No counterfactual needed.']})
    except Exception as e:
        return pd.DataFrame({'Error': [f"Exception in counterfactual generation: {e}"]})

# ------------------------
# Streamlit App Interface
# ------------------------

st.title("⛵ Demurrage Prediction and Suggestions")

st.subheader("📥 Enter Shipping Details")

quantity = st.number_input("Quantity (MT)", min_value=0.0, value=1000.0)
free_time = st.selectbox("Free Time (Hours)", [6, 24, 27])
discharge_rate = st.number_input("Discharge Rate (MT/hr)", min_value=0.0, value=300.0)
demurrage_rate = st.number_input("Demurrage Rate Per Day (USD)", min_value=0.0, value=10000.0)

if st.button("🔍 Predict & Suggest"):
    input_data = {
        'Quantity': quantity,
        'Free_Time_Hours': free_time,
        'Discharge_Rate': discharge_rate,
        'Demurrage_Rate_Per_day': demurrage_rate
    }

    with st.spinner("Analyzing..."):
        result = generate_demurrage_counterfactual(input_data, df, model)
        print(">>> Counterfactual Result:", result)

    st.subheader("🔎 Results")
    if isinstance(result, pd.DataFrame):
        if result.empty:
            st.warning("⚠️ Counterfactual returned empty. Try changing the input values.")
        else:
            st.dataframe(result)
    else:
        st.error("❌ Unexpected error occurred. Check the input format or model.")
