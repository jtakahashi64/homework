# 500
a = int(input())
# 100
b = int(input())
# 50
c = int(input())
goal = int(input())

count = 0

for aI in range(a + 1):
    for bI in range(b + 1):
        for cI in range(c + 1):
            r = aI * 500 + bI * 100 + cI * 50
            if r == goal:
                count += 1

print(count)
