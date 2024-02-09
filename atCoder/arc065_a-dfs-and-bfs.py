import sys
sys.setrecursionlimit(10**6)

target = input()

divides = ["dream", "dreamer", "erase", "eraser"]

# stack = [""]
stack = [0]

can = False

# while(stack != []):
#     # DFS: LIFO: n = len(stack) - 1
#     # BFS: FIFO: n = 0
#     n = 0
#     item = stack[n]
#     del stack[n]

#     if target == item:
#         can = True

#     if (target.startswith(item)):
#         for divide in divides:
#             stack.append(item + divide)

while (stack != []):
    # BFS: n = len(stack) - 1
    # DFS: n = 0
    n = len(stack) - 1
    item = stack[n]
    del stack[n]

    if len(target) == item:
        can = True
        break

    for divide in divides:
        s = item
        e = item + len(divide)
        if target[s:e] == divide:
            stack.append(e)

if can:
    print("YES")
else:
    print("NO")
