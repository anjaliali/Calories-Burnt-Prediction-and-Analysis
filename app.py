import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('xgb.pkl', 'rb'))

# Streamlit app
st.title("Calories Burnt Prediction System")

# Input fields
st.header("Enter Input Details")
gender = st.number_input("Gender (0 for Female, 1 for Male)", min_value=0, max_value=1, step=1)
age = st.number_input("Age", min_value=0, step=1)
height = st.number_input("Height (in cm)", min_value=0.0, step=0.1)
weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1)
duration = st.number_input("Duration (in minutes)", min_value=0.0, step=0.1)
heart_rate = st.number_input("Heart Rate (in bpm)", min_value=0, step=1)
body_temp = st.number_input("Body Temperature (in °C)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    try:
        # Create a DataFrame for the input
        input_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                                  columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"The predicted value is: {round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

