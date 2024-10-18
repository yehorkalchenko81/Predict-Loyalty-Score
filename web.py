import streamlit as st
import pandas as pd
import joblib

random_forest_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Predict customer loyalty score')

age = st.slider('Pick age', 1, 100)
annual_income = st.number_input("Insert annual income per year in USD", step=1000, format='%d')
purchase_amount = st.number_input("Insert purchase ammount per year in USD (Optional expenses)", step=10, format='%d')
purchase_frequency = st.number_input("Insert purchase frequency per year in USD (Optional expenses)", step=1, format='%d')
region = st.radio('Choose region', ['North', 'West', 'South', 'East'])

customer_data = {
    'age': [age],
    'annual_income': [annual_income],
    'purchase_amount': [purchase_amount],
    'region': [region],
    'purchase_frequency': [purchase_frequency]
}

customer_df = pd.DataFrame(customer_data)

directions = {
    'North': 0,
    'East': 1,
    'South': 2,
    'West': 3
}

customer_df['region'] = customer_df['region'].map(directions)

customer_df = scaler.transform(customer_df)

if st.button("Predict loyalty Score"):
    predicted_customer_loyalty = random_forest_model.predict(customer_df)
    st.header(f'Customer Loyalty Score: {predicted_customer_loyalty}')
