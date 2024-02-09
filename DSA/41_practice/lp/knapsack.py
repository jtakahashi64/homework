from pulp import *


def solve_candy_knapsack(all_item_number_max, one_item_number_max, values, spaces, space_min, space_max):
    problem = LpProblem('Knapsack', LpMinimize)

    d_vars = [LpVariable(f'x{i}', lowBound=0, upBound=ki, cat='Integer') for (i, ki) in enumerate(one_item_number_max)]

    value_sum = lpSum([value_i * x_i for (value_i, x_i) in zip(values, d_vars)])

    problem += value_sum

    problem += lpSum(d_vars) <= all_item_number_max

    space_sum = lpSum([space_i * x_i for (space_i, x_i) in zip(spaces, d_vars)])

    problem += space_sum <= space_max
    problem += space_sum >= space_min

    status = problem.solve()

    print(status)

    if status != constants.LpStatusOptimal:
        return

    solution_values = [x.varValue for x in d_vars]

    for (i, solution_i) in enumerate(solution_values):
        print(f'\t{i}: {solution_i}')

    print(f'Value: {sum([(value_i * solution_i) for (value_i, solution_i) in zip(values, solution_values)])}')
    print(f'Space: {sum([(space_i * solution_i) for (space_i, solution_i) in zip(spaces, solution_values)])}')


all_item_number_max = 12
one_item_number_max = [10, 12, 10, 11, 10]
values = [0.2, 0.5, 0.1, 0.4, 0.8]
spaces = [25, 12, 22, 14, 33]
space_min = 250
space_max = 500

solve_candy_knapsack(all_item_number_max, one_item_number_max, values, spaces, space_min, space_max)
