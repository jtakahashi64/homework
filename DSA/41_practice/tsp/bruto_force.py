import itertools

cities = [0, 1, 2, 3]

dist_matrix = [
    [0,  10, 15, 20],
    [10, 0,  35, 25],
    [15, 35, 0,  30],
    [20, 25, 30, 0]
]

min_path = None
min_dist = float('inf')

perms = itertools.permutations(cities)  # すべての順列を生成

for perm in perms:
    current_dist = 0

    for i in range(len(perm)):
        # 最初に戻る必要がない
        if i != len(perm) - 1:
            current_dist += dist_matrix[perm[i]][perm[i+1]]

        # 最初に戻る必要がある
        if i == len(perm) - 1:
            current_dist += dist_matrix[perm[i]][perm[0+0]]

    # 最短経路を更新
    if current_dist < min_dist:
        min_dist = current_dist
        min_path = perm

# 結果を表示
print(min_path)
print(min_dist)
