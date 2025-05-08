import streamlit as st
import joblib
import pandas as pd
import copy

# Load the model
model = joblib.load('model.pkl')

# Function to check demurrage for given input
def check_no_demurrage(input_dict, model):
    test_df = pd.DataFrame([input_dict])
    return model.predict(test_df)[0] == 0

# Main threshold-finding function
def find_discharge_rate_threshold(input_dict, model, min_rate=100, step=-1, min_quantity=1000, quantity_step=-100):
    base_input = copy.deepcopy(input_dict)
    current_rate = base_input['Discharge_Rate']

    # Check if initial input is already avoiding demurrage
    if check_no_demurrage(base_input, model):
        st.success("✅ No demurrage case: The current discharge rate already avoids demurrage.")
        return base_input['Discharge_Rate'], base_input['Quantity']

    # First attempt: vary only discharge rate
    for rate in range(current_rate, min_rate - 1, step):
        base_input['Discharge_Rate'] = rate
        if check_no_demurrage(base_input, model):
            st.success(f"✅ Discharge Rate threshold to avoid demurrage: {rate}")
            return rate, base_input['Quantity']

    # If that fails, try varying quantity
    st.warning("⚠️ No discharge rate alone could avoid demurrage. Trying to adjust quantity as well...")
    original_quantity = input_dict['Quantity']

    for q in range(original_quantity, min_quantity - 1, quantity_step):
        for rate in range(current_rate, min_rate - 1, step):
            modified_input = copy.deepcopy(input_dict)
            modified_input['Quantity'] = q
            modified_input['Discharge_Rate'] = rate
            if check_no_demurrage(modified_input, model):
                st.success(f"✅ Found combination to avoid demurrage: Discharge Rate = {rate}, Quantity = {q}")
                return rate, q

    st.error("❌ No combination of Discharge Rate and Quantity within the given range could avoid demurrage.")
    return None, None

# Streamlit layout
st.title('📦 Demurrage Prediction Tool')
st.subheader('Find the minimum Discharge Rate (and possibly Quantity) needed to avoid demurrage.')

# Input fields
quantity = st.number_input("Quantity", value=6000)
free_time_hours = st.number_input("Free Time (hours)", value=6)
discharge_rate = st.number_input("Discharge Rate", value=400)
demurrage_rate_per_day = st.number_input("Demurrage Rate Per Day (USD)", value=42500)

# Package input
input_data = {
    'Quantity': quantity,
    'Free_Time_Hours': free_time_hours,
    'Discharge_Rate': discharge_rate,
    'Demurrage_Rate_Per_day': demurrage_rate_per_day
}

# Trigger threshold finding
if st.button('🔍 Find Threshold'):
    rate, q = find_discharge_rate_threshold(input_data, model)
    #if rate and q:
        #st.info(f"🧾 Use Discharge Rate: {rate}, Quantity: {q} to avoid demurrage.")

