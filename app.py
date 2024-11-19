import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  # Remove non-alphabetic characters
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text

# Load trained model and tokenizer
model = load_model("disaster_rnn_model_optimized.h5")
tokenizer = Tokenizer()  # Assuming you reinitialize the tokenizer with the training data
word_index = tokenizer.word_index

# App setup
st.title("Disaster Classification App")
st.write("This app classifies text as disaster-related or not disaster-related using an RNN-based model.")

# Input text
user_input = st.text_area("Enter the news or tweet text:")

# Preprocess and predict
max_len = 250  # Same max_len used in training

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

# Display results
if st.button("Predict"):
    if user_input:
        result, score = predict_news(user_input)
        st.write(f"**Result:** {result}")
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.write("Please enter some text to predict.")

