#1
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")
plt.plot(x, softmax(x), label="Softmax")
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
