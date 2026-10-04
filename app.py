import pandas as pd
import streamlit as st

from src.data import load_features
from src.evaluate import get_models, predict_next_day

feat = load_features()
latest = pd.read_csv("data/clean/latest.csv", parse_dates=["Date"])

st.title("Next-day stock return prediction")

ticker = st.selectbox("Ticker", sorted(feat["Ticker"].unique()))
model_name = st.selectbox("Model", list(get_models()))

pred, date = predict_next_day(ticker, model_name, feat, latest)

st.write(f"Based on data up to {date.date()}")
st.metric("Predicted next-day return", f"{pred:.3%}")
st.write("Direction: " + ("Up" if pred > 0 else "Down"))