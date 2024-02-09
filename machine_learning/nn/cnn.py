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


class ReLULayer:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, prev_d_x):
        return prev_d_x * (self.x > 0)


class FlatLayer:
    def forward(self, x):
        self.x = x
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

    def backward(self, prev_d_x):
        return prev_d_x.reshape(self.x.shape)


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


class ConvLayer:
    def __init__(self, lr=0.01):
        self.kernels = np.array([
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        ]).astype(float)
        self.lr = lr

    def forward(self, sources):
        self.sources = sources

        n, h, w = self.sources.shape

        self.kernel_size = self.kernels.shape[0]
        self.square_size = self.kernels.shape[1]

        k = self.kernel_size

        h = h - self.square_size + 1
        w = w - self.square_size + 1

        shape = (n, k, h, w)

        self.x = np.zeros(shape)

        for n_i in range(n):
            for k_i in range(k):
                for h_i in range(h):
                    for w_i in range(w):
                        crop = self.sources[n_i, h_i:h_i +
                                            self.square_size, w_i:w_i+self.square_size]

                        self.x[n_i, k_i, h_i, w_i] = np.sum(
                            self.kernels[k_i] * crop)

        return self.x

    def backward(self, prev_d_x):
        n, k, h, w = prev_d_x.shape

        d_x = np.zeros_like(self.sources).astype(float)
        d_k = np.zeros_like(self.kernels).astype(float)

        for n_i in range(n):
            for k_i in range(k):
                for h_i in range(h):
                    for w_i in range(w):
                        crop = self.sources[n_i, h_i:h_i +
                                            self.square_size, w_i:w_i+self.square_size]

                        d_x[n_i, h_i:h_i+self.square_size, w_i:w_i +
                            self.square_size] += prev_d_x[n_i, k_i, h_i, w_i] * self.kernels[k_i]
                        d_k[k_i] += prev_d_x[n_i, k_i, h_i, w_i] * crop

        self.kernels -= self.lr * d_k

        return d_x


class PoolLayer:
    def __init__(self, pool_size=2, lr=0.01):
        self.pool_size = pool_size
        self.lr = lr

    def forward(self, sources):
        self.sources = sources

        n, k, h, w = sources.shape

        h = h // self.pool_size
        w = w // self.pool_size

        shape = (n, k, h, w)

        self.x = np.zeros(shape)

        self.max_indices = np.zeros_like(self.x, dtype=int)

        for n_i in range(n):
            for k_i in range(n):
                for h_i in range(h):
                    for w_i in range(w):
                        h_start = h_i * self.pool_size
                        w_start = w_i * self.pool_size

                        window = sources[n_i, k_i, h_start:h_start +
                                         self.pool_size, w_start:w_start+self.pool_size]

                        self.x[n_i,
                               k_i,
                               h_i,
                               w_i] = np.max(window)

                        self.max_indices[n_i,
                                         k_i,
                                         h_i,
                                         w_i] = np.argmax(window)
        return self.x

    def backward(self, prev_d_x):
        n, k, h, w = prev_d_x.shape

        d_x = np.zeros_like(self.sources)

        for n_i in range(n):
            for k_i in range(k):
                for h_i in range(h):
                    for w_i in range(w):
                        h_start = h_i * self.pool_size
                        w_start = w_i * self.pool_size

                        h_max, w_max = divmod(
                            self.max_indices[n_i, k_i, h_i, w_i], self.pool_size)

                        d_x[n_i, k_i, h_start+h_max, w_start +
                            w_max] = prev_d_x[n_i, k_i, h_i, w_i]

        return d_x


epochs = 1000
lr = 0.01

i_square_size = 16
o_square_size = 6

images = np.array([
    [
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    ],  # the Cat
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    ],  # the Pot
])

# cat_or_pot_flags is one hot vector
# flag index 0 is the Cat
# flag index 1 is the Pot
cat_or_pot_flags = np.array([
    [1, 0],  # the Cat
    [0, 1],  # the Pot
])

loss_layer = SoftmaxAndCrossEntropyLayer(y=cat_or_pot_flags)
relu_layer = ReLULayer()
flat_layer = FlatLayer()
linear_layer = LinearLayer(lr, np.random.randn(32, 2), np.random.rand(2))
conv_layer = ConvLayer()
pool_layer = PoolLayer()

for epoch in range(epochs):
    x = conv_layer.forward(images)
    x = pool_layer.forward(x)
    x = flat_layer.forward(x)
    x = linear_layer.forward(x)
    l = loss_layer.forward(x)

    if epoch % 10 == 0:
        print(l)

    d = loss_layer.backward()
    d = linear_layer.backward(d)
    d = flat_layer.backward(d)
    d = pool_layer.backward(d)
    d = conv_layer.backward(d)
