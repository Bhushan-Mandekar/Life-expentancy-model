import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load files
# ---------------------------
model = joblib.load("life_expectancy_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🌍 Life Expectancy Prediction (Fixed 212 Features)")

# ---------------------------
# Collect inputs
# ---------------------------
year = st.number_input("Year", 2000, 2035, 2025)
adult_mortality = st.number_input("Adult Mortality", 0.0)
infant_deaths = st.number_input("Infant Deaths", 0.0)
alcohol = st.number_input("Alcohol", 0.0)
percentage_expenditure = st.number_input("Percentage Expenditure", 0.0)
hepatitis_b = st.number_input("Hepatitis B", 0.0)
measles = st.number_input("Measles", 0.0)
bmi = st.number_input("BMI", 0.0)
under_five_deaths = st.number_input("Under Five Deaths", 0.0)
polio = st.number_input("Polio", 0.0)
total_expenditure = st.number_input("Total Expenditure", 0.0)
diphtheria = st.number_input("Diphtheria", 0.0)
hiv_aids = st.number_input("HIV/AIDS", 0.0)
gdp = st.number_input("GDP", 0.0)
population = st.number_input("Population", 0.0)
thinness_1_19 = st.number_input("Thinness 1-19", 0.0)
thinness_5_9 = st.number_input("Thinness 5-9", 0.0)
income = st.number_input("Income Composition", 0.0)
schooling = st.number_input("Schooling", 0.0)

status = st.selectbox("Status", ["Developing", "Developed"])

country_columns = [c for c in feature_names if "Country_" in c]
country = st.selectbox("Country", [c.replace("Country_", "") for c in country_columns])

# ---------------------------
# Predict
# ---------------------------
if st.button("Predict"):

    # Create full feature frame (212 columns)
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Fill numeric values safely
    mapping = {
        "Year": year,
        "Adult Mortality": adult_mortality,
        "infant deaths": infant_deaths,
        "Alcohol": alcohol,
        "percentage expenditure": percentage_expenditure,
        "Hepatitis B": hepatitis_b,
        "Measles ": measles,
        " BMI ": bmi,
        "under-five deaths ": under_five_deaths,
        "Polio": polio,
        "Total expenditure": total_expenditure,
        "Diphtheria ": diphtheria,
        " HIV/AIDS": hiv_aids,
        "GDP": gdp,
        "Population": population,
        " thinness  1-19 years": thinness_1_19,
        " thinness 5-9 years": thinness_5_9,
        "Income composition of resources": income,
        "Schooling": schooling
    }

    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    # Country encoding
    country_col = f"Country_{country}"
    if country_col in input_df.columns:
        input_df[country_col] = 1

    # Status encoding (depends on training)
    if "Status_Developing" in input_df.columns:
        input_df["Status_Developing"] = 1 if status == "Developing" else 0

    # Ensure correct order (VERY IMPORTANT)
    input_df = input_df[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"🌟 Predicted Life Expectancy: {prediction:.2f} Years")