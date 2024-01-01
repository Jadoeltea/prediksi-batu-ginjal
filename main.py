import streamlit as st
from function import load_data
from Tabs import home, predict, visualise

Tabs = {
    "Home": home,
    "Prediction": predict,
    "Visualisation": visualise
}

st.sidebar.title("Navigasi")

page = st.sidebar.radio("Pages", list(Tabs.keys()))

data, x, y = load_data()

if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(data, x, y)
else:
    Tabs[page].app()
