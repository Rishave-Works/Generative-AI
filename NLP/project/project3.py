import streamlit as st
import re
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# DOWNLOAD NLTK DATA
nltk.download("punkt")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT CONFIG
st.set_page_config(page_title="Advanced NLP App", layout="wide")

st.title("Advanced NLP Preprocessing App")
st.write("Regex Cleaning | TF-IDF | Word Embeddings")

# USER INPUT
text = st.text_area(
    "Enter text",
    height=150,
    placeholder="Example: NLP is amazing! It helps machines understand text."
)

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Choose NLP Technique",
    [
        "Regex Text Cleaning",
        "TF-IDF",
        "Word Embeddings"
    ]
)

# PROCESS BUTTON
if st.button("Process"):

    if text.strip() == "":
        st.warning("Please enter some text")

    # ---------------- REGEX CLEANING ----------------
    elif option == "Regex Text Cleaning":
        st.subheader("Text Cleaning using Regular Expression")

        # Lowercase
        text_lower = text.lower()

        # Remove numbers & punctuation using regex
        cleaned_text = re.sub(r"[^a-z\s]", "", text_lower)

        # Tokenization
        tokens = word_tokenize(cleaned_text)

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(tokens))

    # ---------------- TF-IDF ----------------
    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation")

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([text])

        words = tfidf.get_feature_names_out()
        scores = X.toarray()[0]

        df = pd.DataFrame({
            "Word": words,
            "TF-IDF Score": scores
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

    # ---------------- WORD EMBEDDINGS ----------------
    elif option == "Word Embeddings":
        st.subheader("Word Embeddings using Word2Vec")

        # Regex cleaning
        cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        tokens = word_tokenize(cleaned_text)

        # Word2Vec model
        model = Word2Vec(
            [tokens],
            vector_size=50,
            window=5,
            min_count=1,
            workers=4
        )

        # Display word vectors
        vectors = []
        for word in model.wv.index_to_key:
            vectors.append([word] + list(model.wv[word][:5]))

        df = pd.DataFrame(
            vectors,
            columns=["Word", "V1", "V2", "V3", "V4", "V5"]
        )

        st.write("Showing first 5 dimensions of word vectors")
        st.dataframe(df, use_container_width=True)