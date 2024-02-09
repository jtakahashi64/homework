inputAmount = int(input())
numbers = []

for i in range(inputAmount):
    numbers.append(int(input()))

sortedNumbers = sorted(numbers, reverse=True)

hash = {}

for n in sortedNumbers:
    hash[n] = True

print(len(hash.keys()))
