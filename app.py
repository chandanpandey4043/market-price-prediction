# Save this as app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load('market_price_model.pkl')

# Load the dataset to get the original encoders
df = pd.read_csv('market_data.csv')

# Initialize the LabelEncoders
le_dict = {}
for column in df.columns:
    le = LabelEncoder()
    le.fit(df[column])
    le_dict[column] = le

# HTML & CSS for custom navbar and footer
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: space-around;
            background-color: #f8f9fa;
            padding: 1em;
        }
        .navbar a {
            text-decoration: none;
            color: black;
            font-weight: bold;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: black;
            text-align: center;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="navbar">
        <a href="#home">Home</a>
        <a href="#predict">Predict</a>
    </div>
""", unsafe_allow_html=True)

# Home page
if st.checkbox("Home"):
    st.title("Market Price Prediction")
    st.write("This application predicts the market price based on various features.")

# Prediction page
if st.checkbox("Predict"):
    st.header("Predict Market Price")
    
    weather = st.selectbox("Weather Conditions", ["extreme", "normal"])
    pest = st.selectbox("Pest and Disease Outbreaks", ["yes", "no"])
    cost = st.selectbox("Input Costs", ["Increased", "decreased"])
    policy = st.selectbox("Government Policies", ["relaxed", "strict"])
    trade = st.selectbox("Global Markets and Trade", ["Normal", "depression", "growth"])
    demand = st.selectbox("Consumer Demand", ["High", "Low"])
    
    if st.button("Predict"):
        # Encode input data
        input_data = pd.DataFrame({
            "Weather Conditions": [weather],
            "Pest and Disease Outbreaks": [pest],
            "Input Costs": [cost],
            "Government Policies": [policy],
            "Global Markets and Trade": [trade],
            "Consumer Demand": [demand]
        })
        
        for column in input_data.columns:
            le = le_dict[column]
            input_data[column] = le.transform(input_data[column])
        
        prediction = model.predict(input_data)
        st.write(f"Predicted Market Price: {'High' if prediction[0] == 1 else 'Low'}")

# Footer
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Market Prediction. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
