s = input()

count = 0

array = list(s)

if array[0] == "1":
    count += 1
if array[1] == "1":
    count += 1
if array[2] == "1":
    count += 1

print(count)
