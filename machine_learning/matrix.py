import numpy as np

data = np.array([0, 0, 1, 0, 0, 0])

w1 = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
])

w2 = np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
])

x = data

x = np.dot(x, w1)
x = np.dot(x, w2)

print(x)
