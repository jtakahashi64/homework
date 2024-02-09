# def main(pairs, c):
#     # 最大化
#     # memo = [float('-inf')] * (c + 1)
#     # 最小化
#     # memo = [float('+inf')] * (c + 1)
#     memo = [float('-inf')] * (c + 1)
#     used = [-1] * (c + 1)

#     memo[0] = 0

#     for i in range(1, c + 1):
#         # 現在+過去でもっとも大きい/小さい値になるものを採用する
#         for pair in pairs:
#             current_w = pair['w']
#             current_v = pair['v']

#             # i に対象のスペースが収まらない場合はスキップ
#             if i < current_w:
#                 continue

#             r = i - current_w

#             # 最大化
#             # if memo[i] < current_v + memo[r]:
#             # 最小化
#             # if memo[i] > current_v + memo[r]:
#             if memo[i] < current_v + memo[r]:
#                 memo[i] = current_v + memo[r]
#                 used[i] = current_w

#     print(memo)
#     print(used)

#     r = []
#     w = c
#     while w > 0:
#         r.append(used[w])
#         w -= used[w]

#     return memo[c], r

def main(pairs, c):
    n = len(pairs)

    # メモ化テーブル
    # 品物 n(index i) と 重量 c(index j) の 2 次元配列
    memo = [[float('-inf') for _ in range(c + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        memo[i][0] = 0

    for i in range(1, n + 1):
        item_i = i - 1
        current_w = pairs[item_i]['w']
        current_v = pairs[item_i]['v']

        for j in range(1, c + 1):
            if current_w > j:
                memo[i][j] = memo[i - 1][j]
                continue

            r = j - current_w

            memo[i][j] = max(
                memo[i - 1][r] + current_v,
                memo[i - 1][j],
                # memo[i][r] + current_v, # 重複を許す場合
            )

    r = []
    j = c
    for i in range(n, 0, -1):
        if memo[i][j] != memo[i-1][j]:
            item_i = i - 1
            r.append(item_i)
            j -= pairs[item_i]['w']

    return memo[n][c], r


# vの最大化
pairs = [
    {'w': 1, 'v': 2},
    {'w': 2, 'v': 5},
    {'w': 3, 'v': 9},
    {'w': 4, 'v': 6},
]
r = main(pairs, 4)
# (11, [2, 0])
r = print(r)
