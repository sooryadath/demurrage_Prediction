import pandas as pd
import dice_ml
from dice_ml import Dice
import traceback

def generate_demurrage_counterfactual(input_dict, df, model):
    try:
        # Define relevant features
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
        print("DEBUG: Query instance for prediction:", query_instance)

        # Predict class
        pred = model.predict(query_instance)[0]
        print("DEBUG: Model prediction:", pred)

        if pred == 1:
            try:
                dice_exp = exp.generate_counterfactuals(
                    query_instance,
                    total_CFs=1,
                    desired_class=0,
                    features_to_vary=['Quantity', 'Discharge_Rate']
                )
                cf_df = dice_exp.visualize_as_dataframe()
                print("DEBUG: Counterfactual result:", cf_df)
                return cf_df
            except Exception as e:
                print("ERROR in DiCE counterfactual generation:", e)
                print(traceback.format_exc())
                return pd.DataFrame({'Error': [f"Failed to generate counterfactual: {e}"]})
        else:
            print("DEBUG: No demurrage predicted, returning message.")
            return pd.DataFrame({'Message': ['No demurrage predicted. No counterfactual needed.']})
    except Exception as e:
        print("ERROR in generate_demurrage_counterfactual:", e)
        print(traceback.format_exc())
        return pd.DataFrame({'Error': [f"General error: {e}"]})
