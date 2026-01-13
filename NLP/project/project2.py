import streamlit as st
import re
import spacy
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="NLP Feature Extraction", layout="wide")

st.title("NLP Feature Extraction App")
st.write("TF-IDF and Word Embeddings using Regex-based Cleaning")

#USER INPUT 
text = st.text_area(
    "Enter text",
    height=150,
    placeholder="Example: NLP is amazing! It helps machines understand language."
)

option = st.sidebar.radio(
    "Select Feature Extraction Technique",
    ["TF-IDF", "Word Embeddings", "Word2Vec"]
)

#REGEX CLEANING FUNCTION
def regex_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  #remove url
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text) #remove email
    text = re.sub(r"[^a-zA-Z\s]", "", text)  #remove non-alphabet
    text = re.sub(r"\s+", " ", text).strip()  #remove extra space

    # SpaCy stopwords removal
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]

    return " ".join(tokens)


#PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        cleaned_text = regex_clean(text)

        st.markdown("### 🔹 Cleaned Text (Regex) :")
        st.write(cleaned_text)

        #TF-IDF
        if option == "TF-IDF":
            st.subheader("TF-IDF Output")

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([cleaned_text])

            df = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out()
            )

            st.dataframe(df, use_container_width=True)


            # Bar chart (Top words)
            tfidf_scores = df.iloc[0].sort_values(ascending=False)[:10]

            fig, ax = plt.subplots()
            tfidf_scores.plot(kind="bar", ax=ax)
            ax.set_title("Top TF-IDF Words")
            st.pyplot(fig)

        #WORD EMBEDDINGS
        elif option == "Word Embeddings":
            st.subheader("Word Embeddings Output")
            doc = nlp(cleaned_text)
            data = []
            for token in doc:
                if token.has_vector:
                    data.append({
                        "Word": token.text,
                        "Vector Size": token.vector.shape[0],
                        "Vector (first 5 values)": token.vector[:5]
                        })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)


        #WORD2VEC
        elif option == "Word2Vec":
            st.subheader("Word2Vec Embeddings")
            cleaned_text = regex_clean(text)
            # Sentence → Word tokenization
            sentences = sent_tokenize(cleaned_text)
            tokenized_sentences = [word_tokenize(sent) for sent in sentences]

            if len(tokenized_sentences) == 0:
                st.warning("Not enough data to train Word2Vec")
            else:
            # Train Word2Vec model
                model = Word2Vec(
                sentences=tokenized_sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                sg=1   # SkipGram
                )

            # Collect vectors
                data = []
                for word in model.wv.index_to_key:
                    data.append({
                        "Word": word,
                        "Vector Size": model.wv[word].shape[0],
                        "Vector (first 5 values)": model.wv[word][:5]
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                st.info("Word2Vec trained on input text using SkipGram")
    


