import pulp
import numpy as np


def find_next_city(x, n, i):
    for j in range(n):
        if x[i][j].varValue == 1:
            return j


def tsp_mtz_encoding(n, cost_matrix):
    prob = pulp.LpProblem('TSP', pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n, cat='Continuous')

    prob += pulp.lpSum(x[i][j] * cost_matrix[i][j] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += x[i][i] == 0

    # b_0 = [
    #     x[0][1],
    #     x[0][2],
    #     x[0][3],
    # ]

    # b_1 = [
    #     x[1][0],
    #     x[1][2],
    #     x[1][3],
    # ]

    # b_2 = [
    #     x[2][0],
    #     x[2][1],
    #     x[2][3],
    # ]

    # b_3 = [
    #     x[3][0],
    #     x[3][1],
    #     x[3][2],
    # ]

    # e_0 = [
    #     x[1][0],
    #     x[2][0],
    #     x[3][0],
    # ]

    # e_1 = [
    #     x[0][1],
    #     x[2][1],
    #     x[3][1],
    # ]

    # e_2 = [
    #     x[0][2],
    #     x[1][2],
    #     x[3][2],
    # ]

    # e_3 = [
    #     x[0][3],
    #     x[1][3],
    #     x[2][3],
    # ]

    # prob += pulp.lpSum(b_0) == 1
    # prob += pulp.lpSum(b_1) == 1
    # prob += pulp.lpSum(b_2) == 1
    # prob += pulp.lpSum(b_3) == 1

    # prob += pulp.lpSum(e_0) == 1
    # prob += pulp.lpSum(e_1) == 1
    # prob += pulp.lpSum(e_2) == 1
    # prob += pulp.lpSum(e_3) == 1

    # 各都市は出発点もしくは到着点として1回だけ訪れる
    for i in range(n):
        prob += pulp.lpSum([x[i][j] for j in range(n) if i != j]) == 1
    for j in range(n):
        prob += pulp.lpSum([x[i][j] for i in range(n) if i != j]) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # i,j の経路が 1 のとき n * 1
                # i,j の経路が 0 のとき n * 0
                #
                # ex) OK
                # i = 2, j = 1
                # 0 - 2 + 4 * 1 <= 3
                # ex) NG
                # i = 2, j = 1
                # 2 - 0 + 4 * 1 <= 3
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    status = prob.solve()

    for i in range(n):
        print(f'u[{i}] = {u[i].varValue}')

    for i in range(n):
        for j in range(n):
            if i != j and x[i][j].varValue == 1:
                print(f'x[{i},{j}] = {x[i][j].varValue}')

    tour = [0]

    for _ in range(n - 1):
        next_city = find_next_city(x, n, tour[-1])
        if next_city is not None:
            tour.append(next_city)

    return tour


cost_matrix = [
    [0,  2, 9,  10],
    [1,  0, 6,  4],
    [15, 7, 0,  8],
    [6,  3, 12, 0]
]

tour = tsp_mtz_encoding(4, cost_matrix)

print(tour)
