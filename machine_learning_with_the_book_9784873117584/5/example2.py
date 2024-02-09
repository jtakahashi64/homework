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


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1  # 微分 * 微分
        dy = dout * 1  # 微分 * 微分

        return dx, dy


objectA = {
    "price": 100,
    "amount": 2,
}

objectO = {
    "price": 150,
    "amount": 3,
}

tax = 1.1

layer1_mul = MulLayer()
layer2_mul = MulLayer()
layer3_add = AddLayer()
layer4_mul = MulLayer()

l1_arg1 = objectA["price"]
l1_arg2 = objectA["amount"]

l2_arg1 = objectO["price"]
l2_arg2 = objectO["amount"]

l4_arg2 = tax

l1_out1 = layer1_mul.forward(l1_arg1, l1_arg2)
print(l1_out1)
l2_out1 = layer2_mul.forward(l2_arg1, l2_arg2)
print(l2_out1)
l3_out1 = layer3_add.forward(l1_out1, l2_out1)
print(l3_out1)
l4_out1 = layer4_mul.forward(l3_out1, l4_arg2)
print(l4_out1)


d_l4_out1 = 1

d_l4_arg1, d_l4_arg2 = layer4_mul.backward(d_l4_out1)
d_l3_arg1, d_l3_arg2 = layer3_add.backward(d_l4_arg1)
d_l2_arg1, d_l2_arg2 = layer2_mul.backward(d_l3_arg2)
d_l1_arg1, d_l1_arg2 = layer1_mul.backward(d_l3_arg1)

print(d_l1_arg1, d_l1_arg2, d_l2_arg1, d_l2_arg2, d_l4_arg2)
