from pulp import *

problem = LpProblem('example', LpMaximize)

x1 = pulp.LpVariable('x1', -15, 15, "Continuous")
x2 = pulp.LpVariable('x2', -15, 15, "Continuous")
x3 = pulp.LpVariable('x3', -15, 15, "Continuous")
x4 = pulp.LpVariable('x4', -15, 15, "Continuous")
x5 = pulp.LpVariable('x5', -15, 15, "Continuous")

problem += (2 * x1) - (3 * x2) + (1 * x3)

problem += (1 * x1) - (1 * x2) + (1 * x3) <= 5
problem += (1 * x1) - (1 * x2) + (4 * x3) <= 7
problem += (1 * x1) + (2 * x2) - (1 * x3) <= 14
problem += (1 * x3) - (1 * x4) + (1 * x5) <= 7

status = problem.solve()

result = value(problem.objective)

print(result)
print(LpStatus[status])
