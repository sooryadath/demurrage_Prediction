import streamlit as st
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice

# --------------------
# Load model and data
# --------------------
try:
    model = joblib.load('model.pkl')
    df = pd.read_excel('cleaned_data.xlsx')
except Exception as e:
    st.error(f"❌ Failed to load model or data: {e}")
    st.stop()

# ------------------------
# Streamlit App Interface
# ------------------------
st.set_page_config(page_title="Demurrage Prediction Tool", layout="centered", initial_sidebar_state="auto")

st.title("⛵ Demurrage Prediction and Suggestions")
st.subheader("📥 Enter Shipping Details")

quantity = st.number_input("Quantity (MT)", min_value=0.0)
free_time = st.selectbox("Free Time (Hours)", [6, 24, 27])
discharge_rate = st.number_input("Discharge Rate (MT/hr)", min_value=0.0)
demurrage_rate = st.number_input("Demurrage Rate Per Day (USD)", min_value=0.0)

# -----------------------------
# Define Counterfactual Logic
# -----------------------------
def generate_demurrage_counterfactual(input_dict, df, model):
    try:
        feature_cols = ['Quantity', 'Free_Time_Hours', 'Discharge_Rate', 'Demurrage_Rate_Per_day']
        target_col = 'Demurrage_Incurred'

        df_dice = df[feature_cols + [target_col]]
        data_dice = dice_ml.Data(dataframe=df_dice, continuous_features=feature_cols, outcome_name=target_col)
        model_dice = dice_ml.Model(model=model, backend="sklearn")
        exp = Dice(data_dice, model_dice)

        query_instance = pd.DataFrame([input_dict])
        pred = model.predict(query_instance)[0]

        print("Prediction:", pred)
        print("Query Instance:\n", query_instance)

        if pred == 1:
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0,
                features_to_vary=['Quantity', 'Discharge_Rate']
            )
            return dice_exp.visualize_as_dataframe()
        else:
            return pd.DataFrame({'Message': ['No demurrage predicted. No counterfactual needed.']})
    except Exception as e:
        return pd.DataFrame({'Error': [f"❌ Unexpected error occurred. Check the input format or model.\nDetails: {e}"]})

# ------------------------
# Button: Predict & Show
# ------------------------
if st.button("🔍 Predict & Suggest"):
    input_data = {
        'Quantity': quantity,
        'Free_Time_Hours': free_time,
        'Discharge_Rate': discharge_rate,
        'Demurrage_Rate_Per_day': demurrage_rate
    }

    st.write("🧾 Input Summary", input_data)

    with st.spinner("Analyzing..."):
        result = generate_demurrage_counterfactual(input_data, df, model)

    st.subheader("🔎 Results")

    if isinstance(result, pd.DataFrame) and 'Error' in result.columns:
        st.error(result['Error'].iloc[0])
    elif result.empty:
        st.warning("⚠️ No counterfactual could be generated for these inputs.")
    else:
        st.dataframe(result)
