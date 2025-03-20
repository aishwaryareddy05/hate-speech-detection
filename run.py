import streamlit as st
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained models
log_reg_model = joblib.load("log_reg_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
svd = joblib.load("svd_transform.pkl")

# Download NLTK resources if needed
nltk.download("stopwords")
nltk.download("wordnet")

# Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    words = [word for word in text.split() if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in words])
    return text

# Function to predict text category with a threshold
def predict_text_category(text):
    cleaned_text = clean_text(text)
    transformed_text = svd.transform(tfidf_vectorizer.transform([cleaned_text]))
    proba = log_reg_model.predict_proba(transformed_text)
    classes = log_reg_model.classes_

    threshold = 0.6  # Adjusted threshold for Hate Speech
    if proba[0][classes.tolist().index("Hate Speech")] >= threshold:
        return "Hate Speech"
    else:
        return classes[proba.argmax()]

# Streamlit UI
st.title("🛑 Hate Speech Detection")
st.write("Enter text below to check if it's Hate Speech, Offensive Language, or No Hate and Offensive.")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip():
        prediction = predict_text_category(user_input)
        st.subheader(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text.")




