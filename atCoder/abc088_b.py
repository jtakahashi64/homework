_ = int(input())
numbers = [int(s) for s in input().split()]

sortedNumbers = sorted(numbers, reverse=True)

a = 0
b = 0

for i, n in enumerate(sortedNumbers):
    if i % 2 == 0:
        a += n
    else:
        b += n

print(a - b)
