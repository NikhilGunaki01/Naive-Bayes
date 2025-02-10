import streamlit as st
import pickle
import numpy as np

# Load the trained Naïve Bayes model
with open("gaussian_nb_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Naïve Bayes Classifier App")
st.write("Enter the feature values to get a prediction.")

# Example features - Change according to your dataset
feature1 = st.number_input("Feature 1:", format="%.2f")
feature2 = st.number_input("Feature 2:", format="%.2f")
feature3 = st.number_input("Feature 3:", format="%.2f")

# Collect user input and make a prediction
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    if not any(np.isnan(input_data)):  # Check if any input is NaN
        prediction = model.predict(input_data)
        st.success(f"Predicted Class: {prediction[0]}")
    else:
        st.error("Please fill in all feature values to get a prediction.")
