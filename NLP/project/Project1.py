import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt
 
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.stem import PorterStemmer , LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download("punkt")
#nltk.download("stopwords")

#Load spacy model
nlp=spacy.load("en_core_web_sm")

st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)

#App Title
st.title("NLP Preprocessing App")
st.write("Tokenization,Text Cleaning,Stemming,Lemmatization, and Bag Of Words")

#User Input
text=st.text_area("Enter text for NLP preprocessing",height=150,
                  placeholder="Example:Rishave is the HOD of HIT and loves NLP")

#sidebar options
option=st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag Of Words"
    ]
)

#Process Button
if st.button("Process Text"):
    if text.strip() =="":
        st.warning("Please enter some text.")
    #Tokenization
    elif option == "Tokenization":
        st.subheader("Tokenization output")
        col1,col2,col3=st.columns(3)

        #Sentence Tokenization
        with col1:
            st.markdown("### Sentence Tokenization")
            sentences= sent_tokenize(text)
            st.write(sentences)
        #Word Tokenization
        with col2:
            st.markdown("### Word Tokenization")
            word= word_tokenize(text)
            st.write(word) 
        #Character Tokenization
        with col3:
            st.markdown("### Character Tokenization")
            character= list(text)
            st.write(character) 
    
    #Text Cleaning
    elif option == "Text Cleaning":
        st.subheader("Text Cleaning output")

        #Convert text t lowercase
        text_lower=text.lower()

        #Remove Punctuation and numbers
        cleaned_text="".join(ch for ch in text_lower if ch not in string.punctuation and not ch.isdigit())

        #Remove Stopwords using spacy
        doc=nlp(cleaned_text)
        final_words=[token.text for token in doc if not token.is_stop and token.text.strip() != ""]
        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    #Stemming
    elif option =="Stemming":
        st.subheader("Stemming Output")
        words=word_tokenize(text)
        porter=PorterStemmer()
        lancaster=LancasterStemmer()

        #Apply stemming
        porter_stem=[porter.stem(word) for word in words]
        lancaster_stem=[lancaster.stem(word) for word in words]  

        # Comparison table
        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": porter_stem,
            "Lancaster Stemmer": lancaster_stem
        }) 
        st.dataframe(df, use_container_width=True)

    #Lemmatization
    elif option =="Lemmatization":
        st.subheader("Lemmatization Output") 
        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)  

    #Bag Of Words
    elif option =="Bag Of Words":
        st.subheader("Bag Of Words Output") 

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        vocab = vectorizer.get_feature_names_out()
        freq = X.toarray()[0]

        df = pd.DataFrame({
            "Word": vocab,
            "Frequency": freq
        }).sort_values(by="Frequency", ascending=False)

        st.markdown("### BoW Frequency Table")
        st.dataframe(df, use_container_width=True) 

        # PIE CHART (TOP-N WORDS)
        st.markdown("### Word Frequency Distribution (Top 10)")

        top_n = 10
        df_top = df.head(top_n)

        fig, ax = plt.subplots()
        ax.pie(
            df_top["Frequency"],
            labels=df_top["Word"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")  # Makes pie circular

        st.pyplot(fig)    