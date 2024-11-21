from flask import Flask, request, jsonify
import tensorflow
import pickle
import nltk
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'disaster_rnn_model_optimized.h5'
model = tensorflow.keras.models.load_model(MODEL_PATH)
tokenizer = Tokenizer()  
word_index = tokenizer.word_index
max_len = 250  

# Preprocessing function


def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text

def preprocess_input(news):
    words = preprocess_text(news).split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=max_len)
    print(padded_review)
    return padded_review

    
@app.route("/" )
def Hello():
    return jsonify({'title': 'Fucking server is started'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        input_data = request.json
        print(input_data)
        text = input_data.get('text')
        print(text)
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        preprocessed_text = preprocess_input(text)
        print(preprocessed_text)
        input_array = np.array([preprocessed_text])
        # Make predictions
        outputs = model.predict(input_array)
        
        # Unpack outputs (assuming they are in the correct order)
        sentiment, prediction = outputs[0][0], outputs[0][1]  
        print(sentiment , prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000 ,debug=True)
