import itertools


def main(dist_matrix):
    n = len(dist_matrix)

    # メモ
    # [(都市の範囲), 対象の都市] = 距離
    memo = {}

    for i in range(1, n):
        sub_range = [0, i]
        k = (frozenset(sub_range), i)

        # [({0, 1}, 1)] = 20
        # [({0, 2}, 2)] = 42
        # [({0, 3}, 3)] = 35
        memo[k] = dist_matrix[0][i]

    for c in range(2, n):
        # k = 2
        # (1, 2)
        # (1, 3)
        # (2, 3)
        #
        # k = 3
        # (1, 2, 3)
        combs = itertools.combinations(range(1, n), c)

        for comb in combs:
            # (0, 1, 2)
            # (0, 1, 3)
            # (0, 2, 3)
            # (0, 1, 2, 3)
            sub_range = [0] + list(comb)
            k = frozenset(sub_range)

            # i は対象の都市
            for i in comb:
                # 都市の範囲 - 対象の都市
                # (0, 2, 1) - 1 = (0, 2)
                # (0, 1, 2) - 2 = (0, 1)
                p = k - {i}

                # 前回分 + 今回分で最小の距離を求める
                #
                # 前回分:
                #   [(都市の範囲), 対象の都市] = 距離
                # 今回分:
                #   [（前回）対象の都市][（今回）対象の都市] = 距離
                min_dist = float('inf')

                for j in comb:
                    if j == 0 or j == i:
                        continue

                    # p = {0, 2}
                    # i = 1
                    # j = 2
                    # [{0, 2}, 2] = 42 + 30
                    dist = memo[(p, j)] + dist_matrix[j][i]

                    if dist < min_dist:
                        min_dist = dist

                memo[(k, i)] = min_dist

    all_range = range(n)
    k = frozenset(all_range)

    return min(memo[(k, i)] + dist_matrix[i][0] for i in range(1, n))


dist = [
    [0,  20, 42, 35],
    [20, 0,  30, 34],
    [42, 30, 0,  12],
    [35, 34, 12, 0]
]

r = main(dist)

print(r)
