# def satisfied_clauses_size(clauses, truth_assignments):
#     n = len(truth_assignments)
#     m = len(clauses)

#     satisfied_size = 0

#     for clause in clauses:
#         is_clause_satisfied = any([truth_assignments[i-1] if i > 0 else not truth_assignments[-i-1] for i in clause])

#         if is_clause_satisfied:
#             satisfied_size += 1

#     return satisfied_size


# def approx_max_sat(n, clauses):
#     T = True
#     F = False

#     clauses = clauses

#     assign = []

#     num_t = 0
#     num_f = 0

#     for i in range(1, n + 1):
#         (n1, f1, phi1) = assign_and_simplify(clauses, i, T)
#         (n2, f2, phi2) = assign_and_simplify(clauses, i, F)

#         e1 = expectation(phi1) + n1
#         e2 = expectation(phi2) + n2

#         if (e1 >= e2):
#             assign.append(T)
#             num_t += n1
#             num_f += f1
#             clauses = phi1
#         else:
#             assign.append(F)
#             num_t += n2
#             num_f += f2
#             clauses = phi2

#     return (assign, num_t)


def expectation(clauses):
    expect = 0
    for clause in clauses:
        expect += 1.0 - (1.0 / (2.0 ** len(clause)))  # add to the expectation based on the probability clause is true.
    return expect


def assign_and_simplify(clauses, var, val):
    _clauses = []

    num_satisfied = 0  # counter for number of clauses we satisfy due to this assignment
    num_falsified = 0  # counter for number of clauses we falsify due to this assignment

    for clause in clauses:
        in_t_clause_var = (+var) in clause  # var の正のリテラルが clause に含まれているか
        in_n_clause_var = (-var) in clause  # var の負のリテラルが clause に含まれているか

        is_adding_plain = True

        # 正のリテラルが含まれていて、かつ val が T
        # 負のリテラルが含まれていて、かつ val が F
        if (in_t_clause_var and val) or (in_n_clause_var and not val):
            is_adding_plain = False

            # 充足数を増やす
            num_satisfied += 1

        # 正のリテラルが含まれていて、かつ val が F
        # 負のリテラルが含まれていて、かつ val が T
        if (in_t_clause_var and not val) or (in_n_clause_var and val):
            is_adding_plain = False

            # var を含まない clause にする
            _clause = ([i for i in clause if (i != +var and i != -var)])

            # ない
            if len(_clause) == 0:
                num_falsified += 1
            # ある
            else:
                _clauses.append(_clause)

        if is_adding_plain:
            _clauses.append(clause)

    return (num_satisfied, num_falsified, _clauses)


n = 5

# x1 or x3 or not x4
# ...
clauses = [
    [+1, +3, -4],
    [+2, -3, -5],
    [-1, -2, -4],
    [+1, -2],
    [-1, +3, -5],
    [+4, +5]
]

e = expectation(clauses)
print(e)

(n1, f1, c1) = assign_and_simplify(clauses, 3, False)

print(n1)
print(f1)
print(c1)
