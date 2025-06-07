import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Загрузка данных
df = pd.read_csv('Tweets.csv')
tweets = df['text'].values
labels = pd.get_dummies(df['airline_sentiment']).values  # Переводим в one-hot

# Параметры
VOCAB_SIZE = 5000
MAX_LEN = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64

# Создаем и сохраняем токенизатор
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

# Преобразуем текст в последовательности
sequences = tokenizer.texts_to_sequences(tweets)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post')

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Создаем модель
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(LSTM_UNITS),
    Dense(3, activation='softmax')  # 3 класса: negative, neutral, positive
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Сохраняем модель
model.save('sentiment_model.h5')
print("Модель и токенизатор сохранены!")