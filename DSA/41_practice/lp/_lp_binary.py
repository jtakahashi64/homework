from pulp import *


def plan_invite_list(n, m, T_lists, G_lists, scores):
    problem = LpProblem('Cover', LpMinimize)

    d_vars = [LpVariable(f'w_{i}', cat='Binary') for i in range(n)]

    problem += lpSum([s_i * w_i for (s_i, w_i) in zip(scores, d_vars)])

    problem += sum(d_vars) >= m

    for t_i in T_lists:
        problem += lpSum([d_vars[j] for j in t_i]) >= len(t_i) / 4

    for g_i in G_lists:
        problem += lpSum([d_vars[j] for j in g_i]) <= 1

    # solve and get the result
    status = problem.solve()

    if status != constants.LpStatusOptimal:
        return

    solution_values = [x.varValue for x in d_vars]

    for (i, solution_i) in enumerate(solution_values):
        print(f'\t{i}: {solution_i}')

    print(f'{sum(solution_values)}')

    for i, t_i in enumerate(T_lists):
        print(f'{i}: {sum([solution_values[j] for j in t_i])}')
    for i, g_i in enumerate(G_lists):
        print(f'{i}: {sum([solution_values[j] for j in g_i])}')


n = 20
m = 12
T_lists = [[1, 5, 12, 18, 19], [2, 3, 4, 6, 7], [1, 2, 4, 7, 8, 9, 10, 11, 12, 14, 16], [1, 3, 4, 5, 6, 13, 15, 17, 18, 19], [1, 5, 7, 8, 9, 19]]
G_lists = [[1, 5], [5, 19], [4, 7], [4, 12], [4, 19], [4, 18], [3, 4, 15, 19], [4, 7, 18, 2]]
scores = [1, 2, 2, 1, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3.5, 1, 0.6, 0, 1, 8, 8]
plan_invite_list(n, m, T_lists, G_lists, scores)
