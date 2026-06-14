import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X=np.random.randint(50,200,(100,4))
Y=np.random.randint(0,2,100)

scaler=StandardScaler()
X=scaler.fit_transform(X)

model=MLPClassifier(hidden_layer_sizes=(5,),
                    max_iter=500)

model.fit(X,Y)

pred=model.predict(X)

print("Original Output :",Y[:10])
print("Predicted Output:",pred[:10])

plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Cost Function")
plt.show()
