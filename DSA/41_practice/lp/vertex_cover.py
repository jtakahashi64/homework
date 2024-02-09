from pulp import *


def compute_optimal_vertex_cover(n, edge_list):
    problem = LpProblem('Vertex-cover', LpMinimize)

    # d_vars = [LpVariable(f'w_{i}', cat='Binary') for i in range(1, n + 1)]
    d_vars = [LpVariable(f'w_{i}', lowBound=0.0, upBound=1.0, cat='Continuous') for i in range(1, n + 1)]

    problem += lpSum(d_vars)

    for (i, j) in edge_list:
        assert 1 <= i <= n
        assert 1 <= j <= n
        problem += d_vars[i - 1] + d_vars[j - 1] >= 1

    status = problem.solve()

    print(LpStatus[status])

    if status != constants.LpStatusOptimal:
        return

    vertex_cover = [x.varValue for x in d_vars]

    return sum(vertex_cover)


result = compute_optimal_vertex_cover(7, [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 6), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (4, 1), (4, 3),
                                      (4, 5), (4, 7), (5, 3), (5, 4), (5, 6), (5, 7), (6, 2), (6, 3), (6, 5), (6, 7), (7, 4), (7, 5), (7, 6)])

print(result)
