n = int(input())

t = [None] * (10**5 + 1)
x = [None] * (10**5 + 1)
y = [None] * (10**5 + 1)

t[0], x[0], y[0] = [0, 0, 0]

for i in range(n):
    t[i + 1], x[i + 1], y[i + 1] = [int(s) for s in input().split()]

can = True

for i in range(n):
    dt = t[i+1] - t[i]
    dist = abs(x[i+1] - x[i]) + abs(y[i+1] - y[i])
    if dt < dist:
        can = False
    if (dist % 2) != (dt % 2):
        can = False

if can:
    print("Yes")
else:
    print("No")
