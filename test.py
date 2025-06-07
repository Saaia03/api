import requests

API_URL = 'http://localhost:5000'

test_texts = [
    "I love this airline! The service was amazing!",
    "The flight was delayed for 5 hours. Terrible experience.",
    "It was okay, nothing special."
]

for text in test_texts:
    response = requests.post(f'{API_URL}/predict', json={'text': text})
    result = response.json()
    print(f"Текст: {result['text']}")
    print(f"Тональность: {result['sentiment']} (уверенность: {result['confidence']:.2f})")
    print("Распределение вероятностей:")
    for k, v in result['probabilities'].items():
        print(f"  {k}: {v:.4f}")
    print("-" * 50)