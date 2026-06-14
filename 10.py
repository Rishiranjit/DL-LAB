import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Bidirectional,LSTM,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

queries=["hello","order status","refund","bye"]
labels=[0,1,2,3]

responses={
    0:"Hi! How can I help you?",
    1:"Your order is on the way.",
    2:"Refund process started.",
    3:"Goodbye!"
}

tokenizer=Tokenizer()
tokenizer.fit_on_texts(queries)

X=tokenizer.texts_to_sequences(queries)
X=pad_sequences(X)

Y=np.array(labels)

model=Sequential([
    Embedding(len(tokenizer.word_index)+1,10),
    Bidirectional(LSTM(16)),
    Dense(4,activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X,Y,epochs=200,verbose=0)

query=["refund"]

query_seq=tokenizer.texts_to_sequences(query)
query_seq=pad_sequences(query_seq,maxlen=X.shape[1])

pred=np.argmax(model.predict(query_seq))

print("Response:",responses[pred])
