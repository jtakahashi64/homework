class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # 微分 * 微分
        dy = dout * self.x  # 微分 * 微分

        return dx, dy


l1_arg1 = 100
l1_arg2 = 2

l2_arg2 = 1.1

mul_layer1 = MulLayer()
mul_layer2 = MulLayer()

l1_out1 = mul_layer1.forward(l1_arg1, l1_arg2)
l2_arg1 = l1_out1
l2_out1 = mul_layer2.forward(l2_arg1, l2_arg2)

print(l2_out1)

d_l2_out1 = 1

d_l2_arg1, d_l2_arg2 = mul_layer2.backward(d_l2_out1)

print(d_l2_arg1, d_l2_arg2)

d_l1_arg1, d_l1_arg2 = mul_layer1.backward(d_l2_arg1)  # l1_out1 = l2_arg1

print(d_l1_arg1, d_l1_arg2)
