import numpy as np


def softmax(x):
    # Countermeasures against buffer overflow
    pre = x - np.max(x)
    # e ^ pre
    e_x = np.exp(pre)
    sum = e_x.sum(axis=1, keepdims=True)
    return e_x / sum


def cross_entropy(y, x):
    pre = y * np.log(x)
    return -np.sum(pre) / len(y)


class SoftmaxAndCrossEntropyLayer:
    def __init__(self, y):
        self.y = y

    def predict(self, x):
        return softmax(x)

    def forward(self, x):
        self.x = softmax(x)
        return cross_entropy(self.y, self.x)

    def backward(self):
        # Differentiation of Cross Entropy + Softmax
        dX = self.x - self.y
        return dX


class LinearLayer:
    def __init__(self, lr, w, b):
        self.lr = lr
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.w) + self.b

    def backward(self, prev_d_x):
        # Partial derivative with respect to "x"
        dX = np.dot(prev_d_x, self.w.T)
        # Partial derivative with respect to "w"
        dW = np.dot(self.x.T, prev_d_x)
        # Partial derivative with respect to "b"
        dB = np.sum(prev_d_x, axis=0)

        self.w -= self.lr * dW
        self.b -= self.lr * dB

        return dX


np.random.seed(0)

epochs = 1000
lr = 0.01

features = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
])

# cat_or_pot_flags is one hot vector
# flag index 0 is the Cat
# flag index 1 is the Pot
cat_or_pot_flags = np.array([
    [1, 0],  # the Cat
    [0, 1],  # the Pot
])

liner_layer = LinerLayer(lr, np.random.randn(4, 2), np.random.rand(2))
softmax_and_cross_entropy_layer = SoftmaxAndCrossEntropyLayer(cat_or_pot_flags)

for epoch in range(epochs):
    x = liner_layer.forward(features)
    l = softmax_and_cross_entropy_layer.forward(x)

    if epoch % 10 == 0:
        print(l)

    d = softmax_and_cross_entropy_layer.backward()
    d = liner_layer.backward(d)
