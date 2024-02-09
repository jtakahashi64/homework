a = [4, 2, 3, 1, 6, 5]
n = len(a)

for i in range(n):
    for j in range(n - 1 - i):
        current_index = j
        next_index = j + 1

        if a[next_index] < a[current_index]:
            a[current_index], a[next_index] = a[next_index], a[current_index]

print(a)
