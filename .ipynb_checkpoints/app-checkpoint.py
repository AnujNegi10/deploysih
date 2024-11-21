import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec


nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text


model = load_model("disaster_rnn_model_optimized.h5")
tokenizer = Tokenizer()  
word_index = tokenizer.word_index


st.title("Disaster Classification App")
st.write("This app classifies text as disaster-related or not disaster-related using an RNN-based model.")


user_input = st.text_area("Enter the news or tweet text:")


max_len = 250  

def preprocess_input(news):
    words = preprocess_text(news).split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

def predict_news(news):
    preprocessed_text = preprocess_input(news)
    prediction = model.predict(preprocessed_text)[0][0]
    sentiment = 'DisasterRelated' if prediction > 0.5 else 'Not Related'
    return sentiment, prediction


if st.button("Predict"):
    if user_input:
        result, score = predict_news(user_input)
        st.write(f"**Result:** {result}")
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.write("Please enter some text to predict.")

