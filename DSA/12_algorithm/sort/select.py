a = [4, 2, 3, 1, 6, 5]
n = len(a)

for i in range(n):
    current_index = i
    for j in range(i, n):
        if a[current_index] > a[j]:
            current_index = j
    a[i], a[current_index] = a[current_index], a[i]

print(a)
