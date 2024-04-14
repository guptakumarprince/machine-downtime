import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the best model along with preprocessing steps
best_model = joblib.load(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_148 document\model building\best_model_with_preprocessing.pkl")
label_encoder = joblib.load(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_148 document\model building\label_encoder.pkl")

# Function to preprocess input data
def preprocess_input_data(input_):
    # Assuming input_data is a DataFrame with the same structure as your training data
    input_data=input_.copy()
    # Apply any necessary transformations
    input_data['Coolant_Temperature'] = np.exp(input_data['Coolant_Temperature'])
    if 'Cutting' in input_data.columns:
        # Apply any necessary transformations
        input_data['Cutting'] = np.log(input_data['Cutting'].replace(0, 1))  # Handling log(0) case

    # Use the loaded preprocessing pipeline
    processed_data = best_model.named_steps['preprocessing'].transform(input_data)

    return processed_data

# Function to inverse label encoding
def inverse_label_encoding(prediction):
    return label_encoder.inverse_transform(prediction)

def main():
    st.title('Machine Downtime Prediction App')

    # Choose input method (manual or CSV upload)
    input_method = st.radio("Choose Input Method:", ["Manual Input", "CSV File Upload"])

    if input_method == "Manual Input":
        # Input fields for user without min_value and max_value
        hydraulic_pressure = st.number_input('Hydraulic Pressure (bar)', value=50.0)
        coolant_pressure = st.number_input('Coolant Pressure (bar)', value=50.0)
        air_system_pressure = st.number_input('Air System Pressure (bar)', value=50.0)
        coolant_temperature = st.number_input('Coolant Temperature', value=50.0)
        hydraulic_oil_temperature = st.number_input('Hydraulic Oil Temperature (°C)', value=50.0)
        spindle_bearing_temperature = st.number_input('Spindle Bearing Temperature (°C)', value=50.0)
        spindle_vibration = st.number_input('Spindle Vibration (µm)', value=50.0)
        tool_vibration = st.number_input('Tool Vibration (µm)', value=50.0)
        spindle_speed = st.number_input('Spindle Speed (RPM)', value=2500)
        voltage = st.number_input('Voltage (volts)', value=250.0)
        torque = st.number_input('Torque (Nm)', value=50.0)
        cutting_force = st.number_input('Cutting Force (kN)', value=50.0)

        # Predict button
        if st.button('Predict'):
            # Create a DataFrame from the user input
            user_input = pd.DataFrame({
                'Hydraulic_Pressure': [hydraulic_pressure],
                'Coolant_Pressure': [coolant_pressure],
                'Air_System_Pressure': [air_system_pressure],
                'Coolant_Temperature': [coolant_temperature],
                'Hydraulic_Oil_Temperature': [hydraulic_oil_temperature],
                'Spindle_Bearing_Temperature': [spindle_bearing_temperature],
                'Spindle_Vibration': [spindle_vibration],
                'Tool_Vibration': [tool_vibration],
                'Spindle_Speed': [spindle_speed],
                'Voltage': [voltage],
                'Torque': [torque],
                'Cutting': [cutting_force],
            })

            # Preprocess the input data
            processed_input = preprocess_input_data(user_input)

            # Make predictions
            prediction = best_model.predict(processed_input)

            # Inverse label encoding
            inverse_prediction = inverse_label_encoding(prediction)

            # Display the prediction
            st.subheader('Prediction')
            st.write(f'The predicted downtime class is: {inverse_prediction[0]}')

    elif input_method == "CSV File Upload":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read the CSV file
            df_input = pd.read_csv(uploaded_file)

            # Display the uploaded data
            st.subheader('Uploaded Data')
            st.write(df_input)
            

            # Preprocess the input data
            processed_input = preprocess_input_data(df_input)
        
            # Make predictions
            predictions = best_model.predict(processed_input)

            # Inverse label encoding
            inverse_predictions = inverse_label_encoding(predictions)

            # Add predictions to the DataFrame
            df_input['Machine_Downtime_Prediction'] = inverse_predictions

            # Display the predictions
            st.subheader('Predictions')
            st.write(df_input)

if __name__ == '__main__':
    main()
