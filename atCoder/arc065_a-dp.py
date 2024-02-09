target = input()

divides = ["dream", "dreamer", "erase", "eraser"]

dp = [None] * 100010

dp[0] = 1

for i in range(len(target)):
    if (dp[i] == None):
        continue
    for divide in divides:
        s = i
        e = i + len(divide)
        if target[s:e] == divide:
            dp[i + len(divide)] = 1

if dp[len(target)] == 1:
    print("YES")
else:
    print("NO")
