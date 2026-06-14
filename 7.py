import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

classes=['Zero','One','Two','Three','Four',
         'Five','Six','Seven','Eight','Nine']

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=5)

loss,acc=model.evaluate(x_test,y_test)

print("Accuracy:",acc)

plt.imshow(x_test[0],cmap='gray')
plt.axis("off")
plt.show()

pred=model.predict(x_test[:1],verbose=0)

print("Predicted:",classes[np.argmax(pred)])
print("Actual:",classes[y_test[0]])
