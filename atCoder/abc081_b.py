numberSize = int(input())
numbers = [int(i) for i in input().split()]


def divCount(numbers, count):
    if all([i % 2 == 0 for i in numbers]):
        return divCount([i / 2 for i in numbers], count + 1)
    return count


count = divCount(numbers, 0)
print(count)
