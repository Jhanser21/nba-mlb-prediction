import streamlit as st
from models.nba_model import train_nba_model
import pandas as pd

st.title("🏀 NBA & ⚾ MLB Prediction App")

# NBA Prediction
st.header("NBA Game Prediction")
model = train_nba_model()
st.write("Model trained on past games.")

pts = st.number_input("Points", min_value=60, max_value=150, value=100)
ast = st.number_input("Assists", min_value=0, max_value=50, value=25)
reb = st.number_input("Rebounds", min_value=0, max_value=50, value=30)
fg = st.slider("Field Goal %", 0.0, 1.0, 0.45)

pred = model.predict([[pts, ast, reb, fg]])
st.success("Prediction: WIN" if pred[0] == 1 else "Prediction: LOSS")
