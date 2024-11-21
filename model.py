import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import nltk
import re
from gensim.models import Word2Vec


nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text


df = pd.read_csv(r'trainDisaster.csv')


df['text'] = df['text'].fillna('').apply(preprocess_text)
df['keyword'] = df['keyword'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')


df['target'] = df['target'].astype(int)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index


X = tokenizer.texts_to_sequences(df['text'])
y = df['target']


max_len = 250 
X = pad_sequences(X, maxlen=max_len)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


w2v_model = Word2Vec(sentences=df['text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)


embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]


model = Sequential([
    Embedding(input_dim=len(word_index) + 1, 
              output_dim=embedding_dim, 
              input_length=max_len, 
              weights=[embedding_matrix], 
              trainable=True), 
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.3)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


earlystopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)


history = model.fit(
    X_train, y_train,
    epochs=20,  
    batch_size=32,  
    validation_split=0.2,
    callbacks=[earlystopping, reduce_lr]
)


model.save('disaster_rnn_model_optimized.h5')


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


news = "The flooding in the region has caused massive destruction."
result, score = predict_news(news)
print(f'Result: {result}')
print(f'Score: {score:.2f}')