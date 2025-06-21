# app/app.py

import streamlit as st
import joblib
import sys
import os

# ✅ Set correct path to parent so it finds 'utils'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

# Now we can import
from utils.text_cleaning import clean_text

# Load model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, "models", "model_nb.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# Streamlit layout
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article and click **Predict** to see if it's Real or Fake.")

# Input box
user_input = st.text_area("🔍 Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        cleaned_text = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vector)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"**Prediction:** {label}")
