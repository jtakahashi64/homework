import sys
sys.setrecursionlimit(10**6)

target = input()

divides = ["dream", "dreamer", "erase", "eraser"]

# def dig(answer):
#     if answer == target:
#         return True

#     for divide in divides:
#         if target.startswith(answer + divide):
#             if dig(answer + divide):
#                 return True

#     return False


def dig(n):
    if n == len(target):
        return True

    for divide in divides:
        s = n
        e = n + len(divide)
        if target[s:e] == divide and e <= len(target):
            if dig(e):
                return True

    return False


# if dig(""):
if dig(0):
    print("YES")
else:
    print("NO")
