#3
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[2,1],[8,0],[1,3],[7,1]])
y = np.array([[1],[0],[1],[0]])
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative(x):
    return x*(1-x)
np.random.seed(1)
W1 = np.random.rand(2,5)
W2 = np.random.rand(5,1)
cost = []
for i in range(1000):
    h = sig(X@W1)
    o = sig(h@W2)
    e = y-o
    cost.append(np.mean(e**2))
    d2 = e*der(o)
    d1 = d2@W2.T*der(h)
    W2 += h.T@d2*0.1
    W1 += X.T@d1*0.1
print((o>0.5).astype(int))
plt.plot(cost)
plt.show()
