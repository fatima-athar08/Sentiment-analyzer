import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer

# =========================
# Load model + vectorizer
# =========================
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()


# =========================
# SAME preprocessing as notebook
# =========================
def clean_text(text):
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'[^a-zA-Z#]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    words = [w for w in words if len(w) > 3]
    return " ".join(words)


# =========================
# UI
# =========================
st.title("💬 Sentiment Analyzer")
st.write("Analyze text sentiment (Normal / Hate Speech)")

text = st.text_area("Enter text:")

# =========================
# Prediction
# =========================
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)

        if cleaned.strip() == "":
            st.warning("Text is too short or contains no meaningful words after cleaning.")
        else:
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]

            if prediction == 0:
                confidence = round(probability[0] * 100, 2)
                st.success(f"Normal / Non-hate 😊 (Confidence: {confidence}%)")
            else:
                confidence = round(probability[1] * 100, 2)
                st.error(f"Hate / Offensive speech 😠 (Confidence: {confidence}%)")