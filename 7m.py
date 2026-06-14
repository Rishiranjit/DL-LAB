
#7
import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(-1,28,28,1)/255.0
x_test=x_test.reshape(-1,28,28,1)/255.0
m1=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10,activation='softmax')
    
])
m2=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])
for m in [m1,m2]:
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
h1=m1.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))
h2=m2.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))
print("CNN:",m1.evaluate(x_test,y_test)[1])
print("CNN+BN+Dropout:",m2.evaluate(x_test,y_test)[1])
plt.plot(h1.history['val_accuracy'])
plt.plot(h2.history['val_accuracy'])
plt.show()
