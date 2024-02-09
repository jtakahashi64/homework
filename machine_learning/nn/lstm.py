import numpy as np


class SigmoidLayer:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.o = 1 / (1 + np.exp(-x))
        return self.o

    def backward(self, prev_d_x):
        d_x = prev_d_x * self.o * (1 - self.o)
        return d_x


class TanhLayer:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.o = np.tanh(x)
        return self.o

    def backward(self, prev_d_x):
        d_x = prev_d_x * (1 - self.o ** 2)
        return d_x


class LossLayer:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return np.mean((self.x - self.y) ** 2)

    def backward(self):
        return 2 * (self.x - self.y)


class LSTMLayer:
    def __init__(self, lr, c_w, b, s_w):
        self.lr = lr
        self.c_w = c_w
        self.s_w = s_w
        self.b = b

    def forward(self, c_x, s_x):
        self.c_x = c_x
        self.s_x = s_x

        return np.dot(self.c_x, self.c_w) + np.dot(self.s_x, self.s_w) + self.b

    def backward(self, prev_d_x):
        # x の偏微分
        d_c_x = np.dot(prev_d_x, self.c_w.T)
        d_s_x = np.dot(prev_d_x, self.s_w.T)
        # w の偏微分
        d_c_w = np.dot(self.c_x.T, prev_d_x)
        d_s_w = np.dot(self.s_x.T, prev_d_x)
        # b の偏微分
        d_b = np.sum(prev_d_x, axis=0)

        self.c_w -= self.lr * d_c_w
        self.s_w -= self.lr * d_s_w
        self.b -= self.lr * d_b

        return d_c_x, d_s_x


class LSTMCell:
    def __init__(self, lr, i_size, h_size):
        self.lr = lr

        self.l_term_memory = 0.0
        self.s_term_memory = 0.0

        self.sigmoid_cell = SigmoidLayer()
        self.tanh_cell = TanhLayer()

        # 忘却ゲート
        # self.f_gate_r_layer = LSTMLayer(
        #                         lr,
        #                         np.full((i_size, h_size), +1.63, dtype=np.float64),
        #                         np.full((h_size,), +1.62, dtype=np.float64),
        #                         np.full((h_size, h_size), +2.70, dtype=np.float64)
        #                     )
        self.f_gate_r_layer = LSTMLayer(
            lr,
            np.random.randn(i_size, h_size),
            np.random.randn(h_size),
            np.random.randn(h_size, h_size)
        )

        # 入力ゲート
        # self.i_gate_p_layer = LSTMLayer(
        #                         lr,
        #                         np.full((i_size, h_size), +0.94, dtype=np.float64),
        #                         np.full((h_size,), -0.32, dtype=np.float64),
        #                         np.full((h_size, h_size), +1.41, dtype=np.float64)
        #                     )
        self.i_gate_p_layer = LSTMLayer(
            lr,
            np.random.randn(i_size, h_size),
            np.random.randn(h_size),
            np.random.randn(h_size, h_size)
        )
        # self.i_gate_r_layer = LSTMLayer(
        #                         lr,
        #                         np.full((i_size, h_size), +1.65, dtype=np.float64),
        #                         np.full((h_size,), +0.62, dtype=np.float64),
        #                         np.full((h_size, h_size), +2.00, dtype=np.float64)
        #                     )
        self.i_gate_r_layer = LSTMLayer(
            lr,
            np.random.randn(i_size, h_size),
            np.random.randn(h_size),
            np.random.randn(h_size, h_size)
        )

        # 出力ゲート
        # self.o_gate_r_layer = LSTMLayer(
        #                         lr,
        #                         np.full((i_size, h_size), -0.19, dtype=np.float64),
        #                         np.full((h_size,), +0.59, dtype=np.float64),
        #                         np.full((h_size, h_size), +4.38, dtype=np.float64)
        #                     )
        self.o_gate_r_layer = LSTMLayer(
            lr,
            np.random.randn(i_size, h_size),
            np.random.randn(h_size),
            np.random.randn(h_size, h_size)
        )

    def forward(self, c_x):
        s_x = np.full((i_size, h_size), self.s_term_memory, dtype=np.float64)

        # 忘却ゲート
        f_gate_r_x = self.f_gate_r_layer.forward(c_x, s_x)
        f_gate_r_a = self.sigmoid_cell.forward(f_gate_r_x)

        self.l_term_memory *= f_gate_r_a

        # 入力ゲート
        i_gate_r_x = self.i_gate_r_layer.forward(c_x, s_x)
        i_gate_r_a = self.sigmoid_cell.forward(i_gate_r_x)

        i_gate_p_x = self.i_gate_p_layer.forward(c_x, s_x)
        i_gate_p_a = self.tanh_cell.forward(i_gate_p_x)

        self.l_term_memory += i_gate_p_a * i_gate_r_a

        # 出力ゲート
        o_gate_r_x = self.o_gate_r_layer.forward(c_x, s_x)
        o_gate_r_a = self.sigmoid_cell.forward(o_gate_r_x)

        o_gate_p_a = self.tanh_cell.forward(self.l_term_memory)

        self.s_term_memory = o_gate_p_a * o_gate_r_a

        # 四捨五入
        # self.l_term_memory = np.round(self.l_term_memory, 1)
        # self.s_term_memory = np.round(self.s_term_memory, 1)

        return self.l_term_memory, self.s_term_memory

    def backward(self, prev_d_x):
        # 出力ゲート
        d_o_gate_r_a = self.sigmoid_cell.backward(prev_d_x)
        self.o_gate_r_layer.backward(d_o_gate_r_a)

        # 入力ゲート
        d_i_gate_r_a = self.sigmoid_cell.backward(prev_d_x)
        self.i_gate_r_layer.backward(d_i_gate_r_a)

        d_i_gate_p_a = self.tanh_cell.backward(prev_d_x)
        self.i_gate_p_layer.backward(d_i_gate_p_a)

        # 忘却ゲート
        d_f_gate_r_a = self.sigmoid_cell.backward(prev_d_x)
        self.f_gate_r_layer.backward(d_f_gate_r_a)


epochs = 1000
lr = 0.01

i_size = 1
h_size = 1

timeline = [
    # np.array([0.00]),
    # np.array([0.50]),
    # np.array([0.25]),
    # np.array([1.00]),
    np.array([1.00]),
    np.array([0.50]),
    np.array([0.25]),
    np.array([1.00]),
]

# y = np.array([0.0])
y = np.array([1.0])

lstm_cell = LSTMCell(lr, i_size, h_size)
loss_layer = LossLayer()

for epoch in range(epochs):
    for t in timeline:
        l_term_memory, s_term_memory = lstm_cell.forward(t)

        x = s_term_memory

    loss = loss_layer.forward(x, y)

    if epoch % 100 == 0:
        print(loss)
        print(x)

    loss_layer_d = loss_layer.backward()
    lstm_cell_d = lstm_cell.backward(loss_layer_d)
