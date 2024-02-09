amount, goal = [int(s) for s in input().split()]


def search():
    for i in range(amount + 1):
        for j in range(amount + 1 - i):
            # for k in range(amount + 1 - i - j):
            k = amount - i - j
            if i + j + k == amount and goal == i * 10000 + j * 5000 + k * 1000:
                return i, j, k
    return -1, -1, -1


i, j, k = search()

print("{} {} {}".format(i, j, k))
