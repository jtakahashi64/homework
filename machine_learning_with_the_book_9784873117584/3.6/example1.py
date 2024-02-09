from dataset.mnist import load_mnist
import numpy as np
import pickle
import sys
import os
sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x)
    s = np.sum(e)

    return e / s


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (_, _), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, a0):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(a0, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = softmax(a3)

    return z3


x, t = get_data()
network = init_network()

accuracy_count = 0

for i in range(len(x)):
    r = predict(network, x[i])
    p = np.argmax(r)

    if p == t[i]:
        accuracy_count += 1

print("Accuracy:", "{}".format(float(accuracy_count) / len(x)))
