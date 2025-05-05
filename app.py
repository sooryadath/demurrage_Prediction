import streamlit as st
import pandas as pd
import joblib
from dice_ml import Dice
import dice_ml

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="Demurrage Prediction Tool", layout="centered")

# ------------------------
# Load model and data
# ------------------------
@st.cache_data
def load_model():
    model = joblib.load("model.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

model = load_model()
df = load_data()

# ------------------------
# Counterfactual Generator
# ------------------------
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

        if pred == 1:
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0,
                features_to_vary=['Quantity', 'Discharge_Rate']
            )
            result_df = dice_exp.visualize_as_dataframe()

            if result_df is None or result_df.empty:
                return pd.DataFrame({'Message': ['⚠️ No valid counterfactuals could be generated.']})
            return result_df

        else:
            return pd.DataFrame({'Message': ['✅ No demurrage predicted. Counterfactual not required.']})

    except Exception as e:
        return pd.DataFrame({'Error': [f"❌ Error: {str(e)}"]})

# ------------------------
# Streamlit App Interface
# ------------------------
st.title("⛵ Demurrage Prediction and Suggestions")
st.subheader("📥 Enter Shipping Details")

# User input
quantity = st.number_input("Cargo Quantity (in tons)", min_value=100.0, value=5000.0)
free_time = st.number_input("Free Time (in hours)", min_value=0.0, value=6.0)
discharge_rate = st.number_input("Discharge Rate (tons/hour)", min_value=50.0, value=400.0)
demurrage_rate = st.number_input("Demurrage Rate (per day)", min_value=1000.0, value=24500.0)

if st.button("Predict & Suggest Improvements"):
    input_data = {
        "Quantity": quantity,
        "Free_Time_Hours": free_time,
        "Discharge_Rate": discharge_rate,
        "Demurrage_Rate_Per_day": demurrage_rate
    }

    with st.spinner("Analyzing..."):
        result = generate_demurrage_counterfactual(input_data, df, model)
        st.subheader("🔎 Results")

        if not isinstance(result, pd.DataFrame):
            st.error("❌ Unexpected output format.")
        elif 'Error' in result.columns:
            st.error(result['Error'].iloc[0])
        elif 'Message' in result.columns:
            st.info(result['Message'].iloc[0])
        else:
            st.success("✅ Counterfactual generated successfully!")
            st.dataframe(result)
