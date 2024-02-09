max, digitsSumMin, digitsSumMax = [int(s) for s in input().split()]


def digitsSum(n):
    sum = 0
    while (n > 0):
        sum += n % 10
        n = int(n / 10)
    return sum


sum = 0

for i in range(0, max + 1):
    r = digitsSum(i)
    if digitsSumMin <= r and r <= digitsSumMax:
        sum += i

print(sum)
