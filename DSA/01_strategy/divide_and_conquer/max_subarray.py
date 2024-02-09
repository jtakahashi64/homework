# def max_subarray(a):
#     n = len(a)
#     max_sum = a[0]

#     for s in range(n):
#         for e in range(s, n):
#             sum = sum(a[s:e+1])
#             max_sum = max(max_sum, sum)

#     return max_sum

# def max_subarray(a):
#     max_alltime = a[0]
#     max_current = a[0]

#     for i in range(1, len(a)):
#         # 今までの合計値と、今回の値を比較して、大きい方を選択する
#         max_current = max(a[i], max_current + a[i])
#         max_alltime = max(max_alltime, max_current)

#     return max_alltime

def max_subarray(a, l, r):
    print("l", l)
    print("r", r)

    if l == r:
        return a[l]

    m = (l + r) // 2

    max_l = max_subarray(a, l,     m)
    print("max_l", max_l)
    max_r = max_subarray(a, m + 1, r)
    print("max_r", max_r)

    sum_l = float('-inf')
    temp_l = 0
    sum_r = float('-inf')
    temp_r = 0

    for i in range(m, l - 1, -1):
        print("p", i)
        temp_l += a[i]
        sum_l = max(sum_l, temp_l)

    for i in range(m + 1, r + 1):
        print("n", i)
        temp_r += a[i]
        sum_r = max(sum_r, temp_r)

    sum = sum_l + sum_r

    print("o_max_l", max_l)
    print("o_max_r", max_r)
    print("o_sum_l", sum_l)
    print("o_sum_r", sum_r)
    print("o_sum", sum)

    return max(max_l, max_r, sum)


a = [100, -2, 5, 10, 11, -4, 15, 9, 18, -2, 21, -11]
n = len(a)

r = max_subarray(a, 0, n - 1)

print(r)
