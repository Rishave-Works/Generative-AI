# file name: sample_app.py
import streamlit as st
import spacy
import nltk

# NLTK example
from nltk.tokenize import word_tokenize

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit app
st.title("Rishav's NLP & Streamlit Sample App")
st.write("Ye app dikhata hai ki Streamlit, spaCy aur NLTK sath me kaam karte hain.")

# User input
user_text = st.text_area("Apna text yahan type karo:")

if user_text:
    st.subheader("NLTK Tokenization:")
    tokens = word_tokenize(user_text)
    st.write(tokens)
    
    st.subheader("spaCy Named Entity Recognition (NER):")
    doc = nlp(user_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        st.write(entities)
    else:
        st.write("Koi entity nahi mili.")

st.write("✅ App ready! Text type karo aur NLP output dekho.")
