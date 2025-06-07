from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Загрузка модели и токенизатора
model = load_model('sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 100
CLASSES = ['negative', 'neutral', 'positive']

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    # Токенизация и подготовка текста
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Предсказание
    prediction = model.predict(padded)[0]
    predicted_class = CLASSES[np.argmax(prediction)]
    
    return jsonify({
        'text': text,
        'sentiment': predicted_class,
        'confidence': float(np.max(prediction)),
        'probabilities': {
            'negative': float(prediction[0]),
            'neutral': float(prediction[1]),
            'positive': float(prediction[2])
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)