import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# CONFIG
# -------------------------------
max_features = 10000
max_len = 200

st.set_page_config(page_title="IMDB Sentiment Analysis")

# -------------------------------
# TRAIN MODEL (cached)
# -------------------------------
@st.cache_resource
def train_model():

    (X_train, y_train), _ = imdb.load_data(num_words=max_features)

    X_train = sequence.pad_sequences(X_train, maxlen=max_len)

    model = Sequential([
        Embedding(max_features, 128, input_length=max_len),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=2
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return model

model = train_model()

# -------------------------------
# WORD INDEX
# -------------------------------
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded], maxlen=max_len)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("LSTM-based Sentiment Classification")

review = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if review == "":
        st.warning("Please enter a review")
    else:
        encoded = encode_review(review)
        prediction = model.predict(encoded)[0][0]

        if prediction > 0.5:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

        st.write("Prediction Score:", round(prediction, 3))
