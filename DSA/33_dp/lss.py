def main(a):
    n = len(a)

    if n == 0:
        return 0

    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if abs(a[i] - a[j]) <= 1:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


r = main([1, 4, 2, -2, 0, -1, 2, 3])

print(r)
