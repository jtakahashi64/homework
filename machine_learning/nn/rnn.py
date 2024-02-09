import numpy as np


class RecurrentLinearLayer:
    def __init__(self, lr, c_w, b, p_w):
        self.lr = lr
        self.c_w = c_w
        self.p_w = p_w
        self.b = b

    def forward(self, c_x, p_x):
        self.c_x = c_x
        self.p_x = p_x

        return np.dot(self.c_x, self.c_w) + np.dot(self.p_x, self.p_w) + self.b

    def backward(self, prev_d_x):
        # Partial derivative with respect to "x"
        d_c_x = np.dot(prev_d_x, self.c_w.T)
        d_p_x = np.dot(prev_d_x, self.p_w.T)
        # Partial derivative with respect to "w"
        d_c_w = np.dot(self.c_x.T, prev_d_x)
        d_p_w = np.dot(self.p_x.T, prev_d_x)
        # Partial derivative with respect to "b"
        d_b = np.sum(prev_d_x, axis=0)

        self.c_w -= self.lr * d_c_w
        self.p_w -= self.lr * d_p_w
        self.b -= self.lr * d_b

        return d_c_x, d_p_x


class LinearLayer:
    def __init__(self, lr, c_w, b):
        self.lr = lr
        self.c_w = c_w
        self.b = b

    def forward(self, c_x):
        self.c_x = c_x

        return np.dot(self.c_x, self.c_w) + self.b

    def backward(self, prev_d_x):
        # Partial derivative with respect to "x"
        d_c_x = np.dot(prev_d_x, self.c_w.T)
        # Partial derivative with respect to "w"
        d_c_w = np.dot(self.c_x.T, prev_d_x)
        # Partial derivative with respect to "b"
        d_b = np.sum(prev_d_x, axis=0)

        self.c_w -= self.lr * d_c_w
        self.b -= self.lr * d_b

        return d_c_x


class ReLULayer:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, prev_d_x):
        return prev_d_x * (self.x > 0)


class LossLayer:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return np.mean((self.x - self.y) ** 2)

    def backward(self):
        return 2 * (self.x - self.y)


epochs = 100
lr = 0.01

i_size = 1
h_size = 2

timeline = [
    np.array([0.25]),  # Day 1 eating value
    np.array([0.50]),  # Day 2 eating value
    np.array([0.75]),  # Day 3 eating value
    np.array([1.00]),  # Day 4 eating value
]

y = np.array([1.25])  # Day 5 eating value

recurrent_linear_layer = RecurrentLinearLayer(
    lr,
    np.random.randn(i_size, h_size),
    np.random.randn(h_size),
    np.random.randn(h_size, h_size)
)
linear_layer = LinearLayer(
    lr,
    np.random.randn(h_size, i_size),
    np.random.randn(i_size)
)
relu_layer = ReLULayer()
loss_layer = LossLayer()

p_x = np.full((i_size, h_size), 0, dtype=np.float64)

for epoch in range(epochs):
    for t in timeline:
        x1 = recurrent_linear_layer.forward(t, p_x)
        x2 = relu_layer.forward(x1)
        p_x = x2

    x3 = linear_layer.forward(x2)

    l1 = loss_layer.forward(x3, y)

    # loss
    print(l1)

    # prediction
    print(x3)

    d1 = loss_layer.backward()
    d2 = linear_layer.backward(d1)
    d3 = relu_layer.backward(d2)
    d4 = recurrent_linear_layer.backward(d3)
