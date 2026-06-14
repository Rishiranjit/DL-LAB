#9 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

text = [
    "i love deep learning",
    "i love machine learning",
    "deep learning is powerful",
    "machine learning is amazing",
    "i enjoy learning python",
    "python is easy to learn"
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
X, y = [], []
for s in seq:
    for i in range(1, len(s)):
        X.append(s[:i])
        y.append(s[i])
X = pad_sequences(X)
y = to_categorical(y)
model = Sequential([
    Embedding(50, 5),
    SimpleRNN(10),
    Dense(y.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200)
test = tokenizer.texts_to_sequences(["love"])[0]
test = pad_sequences([test], maxlen=X.shape[1])
print(tokenizer.index_word[np.argmax(model.predict(test))])
