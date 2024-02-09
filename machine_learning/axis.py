import numpy as np

a = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print(a.shape)
print(a.ndim)

print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.sum(a, axis=2))
print(np.sum(a, axis=-1))
