#4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
X = np.random.randint(50,200,(10,3))
y = np.array([1,0,1,0,1,0,1,0,1,0])
X = MinMaxScaler().fit_transform(X)
model = MLPClassifier(hidden_layer_sizes=(5,),max_iter=500)
model.fit(X,y)
pred = model.predict(X)
print("Original:",y)
print("Predicted:",pred)
plt.plot(model.loss_curve_)
plt.show()
