#After tokenizing text, display a bar chart showing the
#frequency of the top 5 most common words (excluding punctuation).

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import string

st.title("📊 Top 5 Word Frequency")

text = st.text_area("Enter text")

if st.button("Show Frequency Chart"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        tokens = word_tokenize(text.lower())

        stop_words = set(stopwords.words('english'))

        clean_tokens = [
            t for t in tokens
            if t.isalpha() and t not in stop_words
        ]

        freq = Counter(clean_tokens)
        top_words = freq.most_common(5)

        if not top_words:
            st.info("No valid words to display")
        else:
            df = pd.DataFrame(top_words, columns=["Word", "Count"]).set_index("Word")
            st.bar_chart(df)

