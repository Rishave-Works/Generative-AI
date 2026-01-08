import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
st.title("🚫⛔ Stopword removal")
text=st.text_area("Enter text here")
if st.button("Remove stopwords"):
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    filtered=[word for word in tokens if word.lower() not in stop_words and word not in punctuations and not word.isdigit()]
    st.write("✅ Cleaned Tokens:")
    st.write(filtered)
    removed_count=len(tokens)-len(filtered)
    st.write(f"❌ Removed words count: {removed_count}")