import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('logistic_regression_model.pkl')

# Define the app
def main():
    st.title("Crime Data Suspect Prediction")

    # Create input fields for user to enter crime data
    dr_number = st.text_input("DR Number")
    date_reported = st.date_input("Date Reported")
    date_occurred = st.date_input("Date Occurred")
    time_occurred = st.time_input("Time Occurred")
    area_id = st.number_input("Area ID", step=1)
    area_name = st.selectbox("Area Name", options=['Olympic', 'Southeast', 'Northeast', 'Foothill', 'Mission'])
    crime_code = st.number_input("Crime Code", step=1)
    crime_code_description = st.text_input("Crime Code Description")
    victim_age = st.number_input("Victim Age", step=1)
    victim_sex = st.selectbox("Victim Sex", options=['M', 'F', 'Unknown'])
    victim_descent = st.selectbox("Victim Descent", options=['W', 'H', 'B', 'A', 'O', 'Unknown'])
    premise_code = st.number_input("Premise Code", step=1)
    premise_description = st.text_input("Premise Description")
    weapon_used_code = st.number_input("Weapon Used Code", step=1)
    weapon_description = st.text_input("Weapon Description")
    status_code = st.text_input("Status Code")
    status_description = st.selectbox("Status Description", options=['Invest Cont', 'Adult Arrest', 'Juv Arrest', 'Unknown'])
    address = st.text_input("Address")
    location = st.text_input("Location")
    day_of_week_occurred = st.selectbox("Day of Week Occurred", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    part_of_day = st.selectbox("Part of Day", options=['Morning', 'Afternoon', 'Evening', 'Night'])
    crime_severity = st.slider("Crime Severity", 1, 5, 3)

    # Button to make prediction
    if st.button("Predict Suspect Status"):
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame({
            'DR Number': [dr_number],
            'Date Reported': [pd.to_datetime(date_reported)],
            'Date Occurred': [pd.to_datetime(date_occurred)],
            'Time Occurred': [time_occurred.hour * 100 + time_occurred.minute],
            'Area ID': [area_id],
            'Area Name': [area_name],
            'Crime Code': [crime_code],
            'Crime Code Description': [crime_code_description],
            'Victim Age': [victim_age],
            'Victim Sex': [victim_sex],
            'Victim Descent': [victim_descent],
            'Premise Code': [premise_code],
            'Premise Description': [premise_description],
            'Weapon Used Code': [weapon_used_code],
            'Weapon Description': [weapon_description],
            'Status Code': [status_code],
            'Status Description': [status_description],
            'Address': [address],
            'Location': [location],
            'Day of Week Occurred': [day_of_week_occurred],
            'Part of Day': [part_of_day],
            'Crime Severity': [crime_severity]
        })

        # Make prediction
        prediction = model.predict(input_data)
        result = 'Suspect' if prediction[0] == 1 else 'Not Suspect'
        st.success(f"The predicted status is: {result}")

if __name__ == "__main__":
    main()
