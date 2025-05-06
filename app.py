import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Function to find discharge rate threshold
def find_discharge_rate_threshold(input_dict, model, min_rate=100, step=-1):
    import copy

    base_input = copy.deepcopy(input_dict)
    current_rate = base_input['Discharge_Rate']
    
    
    # Check if the input data itself already results in no demurrage (i.e., prediction is 0)
    test_df = pd.DataFrame([base_input])
    initial_pred = model.predict(test_df)[0]
    if initial_pred == 0:
        st.success("✅ No demurrage case: The current discharge rate already avoids demurrage.")
        return current_rate
    if current_rate < min_rate:
        st.error("❌ Starting rate is already lower than min_rate. No need to find threshold.")
        return None

    for rate in range(current_rate, min_rate - 1, step):
        base_input['Discharge_Rate'] = rate
        test_df = pd.DataFrame([base_input])
        pred = model.predict(test_df)[0]
        if pred == 0:
            st.success(f"✅ Discharge Rate threshold to avoid demurrage: {rate}")
            return rate

    st.error("❌ No discharge rate within the given range could avoid demurrage.")
    return None

# Streamlit app layout
st.title('Demurrage Prediction')
st.subheader('Input the data to find the discharge rate threshold.')

# Collect inputs
quantity = st.number_input("Quantity", value=6000)
free_time_hours = st.number_input("Free Time (hours)", value=6)
discharge_rate = st.number_input("Discharge Rate", value=400)
demurrage_rate_per_day = st.number_input("Demurrage Rate Per Day (USD)", value=42500)

input_data = {
    'Quantity': quantity,
    'Free_Time_Hours': free_time_hours,
    'Discharge_Rate': discharge_rate,
    'Demurrage_Rate_Per_day': demurrage_rate_per_day
}

# Button to trigger prediction
if st.button('Find Discharge Rate Threshold'):
    find_discharge_rate_threshold(input_data, model)
    
