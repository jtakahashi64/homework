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


def to_one_hot(id, vocab_size):
    x = np.zeros(vocab_size)
    x[id] = 1
    return x


# Skip-Gram


def generate_training_data(tokens, window_size):
    x, y = [], []

    tokens_size = len(tokens)

    for i in range(tokens_size):
        s = max(0,           i - window_size)
        e = min(tokens_size, i + window_size + 1)

        for j in range(s, e):
            if i == j:
                continue

            x.append(to_one_hot(word_to_id[tokens[i]], len(word_to_id)))
            y.append(to_one_hot(word_to_id[tokens[j]], len(word_to_id)))

    return np.array(x), np.array(y)


np.random.seed(0)

epochs = 10000
lr = 0.01
embed_size = 2
window_size = 1

tokens = []

word_to_id = {}
id_to_word = {}

text = "Cat is cute Cat is cool Cat is good Pot is cute Pot is cool Pot is good"

tokens = text.split()

words = set(tokens)

words_size = len(words)

for i, word in enumerate(words):
    word_to_id[word] = i
    id_to_word[i] = word

x0, y0 = generate_training_data(tokens, window_size)

w_to_e_liner_layer = LinearLayer(lr, np.random.randn(
    words_size, embed_size), np.random.rand(embed_size))
e_to_w_liner_layer = LinearLayer(lr, np.random.randn(
    embed_size, words_size), np.random.rand(words_size))
softmax_and_cross_entropy_layer = SoftmaxAndCrossEntropyLayer(y0)

for epoch in range(epochs):
    x1 = w_to_e_liner_layer.forward(x0)
    x2 = e_to_w_liner_layer.forward(x1)

    l1 = softmax_and_cross_entropy_layer.forward(x2)

    if epoch % 100 == 0:
        print(l1)

    d1 = softmax_and_cross_entropy_layer.backward()
    d2 = e_to_w_liner_layer.backward(d1)
    d3 = w_to_e_liner_layer.backward(d2)

print(word_to_id)
print(w_to_e_liner_layer.w)  # Cat and Pot are close to each other
