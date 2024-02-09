from pulp import *

problem = LpProblem('example', LpMaximize)

x1 = pulp.LpVariable('x1', -1, 1, "Continuous")
x2 = pulp.LpVariable('x2', -1, 1, "Continuous")
x3 = pulp.LpVariable('x3', -1, 1, "Continuous")

problem += (2 * x1) - (3 * x2) + (1 * x3)

problem += (1 * x1) - (1 * x2) >= 0.50
problem += (1 * x1) - (1 * x2) <= 0.75
problem += (1 * x2) - (1 * x3) <= 1.25
problem += (1 * x2) - (1 * x3) >= 0.95

status = problem.solve()

result = value(problem.objective)

print(result)
print(LpStatus[status])
