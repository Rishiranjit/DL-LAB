import numpy as np
import matplotlib.pyplot as plt

X=np.array([[5,1],[8,2],[2,0],[9,3]])
Y=np.array([[0],[1],[0],[1]])

W1=np.random.rand(2,5)
W2=np.random.rand(5,1)

cost=[]

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(1000):

    H=sigmoid(np.dot(X,W1))
    O=sigmoid(np.dot(H,W2))

    error=Y-O
    cost.append(np.mean(error**2))

    dO=error*O*(1-O)
    dH=dO.dot(W2.T)*H*(1-H)

    W2+=0.1*H.T.dot(dO)
    W1+=0.1*X.T.dot(dH)

pred=(O>0.5).astype(int)

print("Final Predictions:")
print(pred)

plt.plot(cost)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()
