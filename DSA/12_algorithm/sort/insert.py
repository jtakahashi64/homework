a = [4, 2, 3, 1, 6, 5]
n = len(a)

for i in range(1, n):
    current_value = a[i]

    prev_index = i - 1

    while 0 <= prev_index and current_value < a[prev_index]:
        a[prev_index + 1] = a[prev_index]
        prev_index -= 1

    a[prev_index + 1] = current_value

print(a)
