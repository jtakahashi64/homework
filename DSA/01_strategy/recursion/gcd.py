def extended_euclidean(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_euclidean(b % a, a)
        print(g, a, x, b, y)
        return (g, y - (b // a) * x, x)


# Using the Extended Euclidean Algorithm to find the Bezout coefficients for 10 and 6
g, x, y = extended_euclidean(3, 20)
print(g, x, y)
