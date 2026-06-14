#10
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["refund issue", "order delay", "payment failed"]
labels = [0, 1, 2]
responses = {
    0: "Refund will be processed",
    1: "Order will arrive soon",
    2: "Try payment again"
}
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts))
y = np.array(labels)
model = Sequential([
    Embedding(20, 5),
    Bidirectional(LSTM(5)),
    Dense(3, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam'
)
model.fit(X, y, epochs=100)
test = pad_sequences(
    tokenizer.texts_to_sequences(["payment issue"]),
    maxlen=X.shape[1]
)
pred = np.argmax(model.predict(test))
print("Bot Reply:", responses[pred])
