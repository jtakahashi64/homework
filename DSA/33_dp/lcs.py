def main(s1, s2):
    m = len(s1)
    n = len(s2)

    memo = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            item_i = i - 1
            item_j = j - 1
            if s1[item_i] == s2[item_j]:
                memo[i][j] = memo[i - 1][j - 1] + 1
            else:
                memo[i][j] = max(memo[i - 1][j], memo[i][j - 1])

    i, j = m, n
    r = []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            r.append(s1[i - 1])
            i -= 1
            j -= 1
        elif memo[i - 1][j] > memo[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(r))


s1 = "hello"
s2 = "hallo"

r = main(s1, s2)

print(r)
