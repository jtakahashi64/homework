a, b = [int(s) for s in input().split()]

product = a * b

if product % 2 == 0:
    print("Even")
else:
    print("Odd")
