#2
import numpy as np
X = np.array([0.8, 0.7])
W1 = np.array([[0.5, 0.2, 0.3],
               [0.4, 0.6, 0.1]])
W2 = np.array([[0.7, 0.2, 0.1],
               [0.3, 0.8, 0.2],
               [0.5, 0.4, 0.6]])
bias = 0.1
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
hidden = relu(np.dot(X, W1) + bias)
output = softmax(np.dot(hidden, W2))
grades = ["Grade A", "Grade B", "Grade C"]
print("Hidden Output:", hidden)
print("Probabilities:", output)
print("Prediction:", grades[np.argmax(output)])
