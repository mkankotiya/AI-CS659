# Program: Random k-SAT Formula Generator
#  What it does:
#   → Creates a random logical formula used in SAT problems
#   → Every clause has exactly 'k' literals (variable or NOT variable)
#   → Formula is made of 'm' clauses and 'n' total variables
#
#  Inputs:
#     k = number of literals in each clause
#     m = number of clauses in the whole formula
#     n = total number of variables (x1, x2, ... xn)
#
#  Example (3-SAT example):
#     Suppose k=3, m=2, n=3
#     Random formula might look like:
#         (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3)
#     Each clause has 3 literals → this is a 3-SAT problem.

import random

def make_random_kSAT(k, m, n):
    # List of variable (x1,x2,...,xn)
    variables = [f"x{i+1}" for i in range(n)]
    formula = []

    for clause_no in range(m):
        # Randomly choose k distinct variables for this clause
        chosen_vars = random.sample(variables, k)
        clause = []

        for v in chosen_vars:
            # Randomly decide variable value true/false
            if random.choice([True, False]):
                clause.append(v)
            else:
                clause.append("¬" + v)  

        # Join with OR 
        clause_str = "(" + " ∨ ".join(clause) + ")"
        formula.append(clause_str)

    # Join clauses with AND 
    sat_formula = " ∧ ".join(formula)
    return sat_formula


if __name__ == "__main__":
    print("Random k-SAT Formula Generator")
    k = int(input("Enter k (number of literals per clause): "))
    m = int(input("Enter m (number of clauses): "))
    n = int(input("Enter n (number of variables): "))

    result = make_random_kSAT(k, m, n)
    print("\nGenerated Random k-SAT formula:\n")
    print(result)
