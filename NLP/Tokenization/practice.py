import streamlit as st
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download("punkt_tab")

st.title("🔍 Text Tokenizer")

text = st.text_area("Enter some text:")
if st.button("Tokenize"):
    tokens = word_tokenize(text)
    st.write("Number of tokens:", len(tokens))
    st.write("Tokens:", tokens)

    token_freq = Counter(tokens)
    st.write("Most frequent tokens:", token_freq.most_common(3))
