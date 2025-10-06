# ----------------------------------------------------------
# Compare Search Algorithms on Random 3-SAT Problems
# ----------------------------------------------------------
# Each SAT formula has m clauses and n variables (x1...xn)
# Algorithms: Hill-Climbing, Beam-Search, Variable-Neighborhood Descent
# Two Heuristics:
#   h1 = number of satisfied clauses
#   h2 = percentage of satisfied clauses
# ----------------------------------------------------------

import random

# Generate random 3-SAT formula
def make_3SAT(m, n):
    variables = [f"x{i+1}" for i in range(n)]
    formula = []
    for _ in range(m):
        chosen_vars = random.sample(variables, 3)
        clause = []
        for v in chosen_vars:
            if random.choice([True, False]):
                clause.append(v)
            else:
                clause.append("¬" + v)
        formula.append(clause)
    return formula


# Evaluate clauses 
def count_satisfied(formula, assignment):
    count = 0
    for clause in formula:
        if any(
            (lit[0] != "¬" and assignment[lit] == True)
            or (lit[0] == "¬" and assignment[lit[1:]] == False)
            for lit in clause
        ):
            count += 1
    return count


# Heuristic functions
def heuristic1(formula, assign):
    return count_satisfied(formula, assign)

def heuristic2(formula, assign):
    return count_satisfied(formula, assign) / len(formula)


# Hill Climbing
def hill_climb(formula, vars_list, heuristic):
    current = {v: random.choice([True, False]) for v in vars_list}
    best_score = heuristic(formula, current)
    improved = True

    while improved:
        improved = False
        for v in vars_list:
            neighbor = current.copy()
            neighbor[v] = not neighbor[v]
            score = heuristic(formula, neighbor)
            if score > best_score:
                current = neighbor
                best_score = score
                improved = True
    return best_score


# Beam Search
def beam_search(formula, vars_list, heuristic, beam_width):
    beam = [{v: random.choice([True, False]) for v in vars_list} for _ in range(beam_width)]

    for _ in range(20): 
        all_neighbors = []
        for a in beam:
            for v in vars_list:
                n = a.copy()
                n[v] = not n[v]
                all_neighbors.append(n)
        all_neighbors.sort(key=lambda x: heuristic(formula, x), reverse=True)
        beam = all_neighbors[:beam_width]
    return heuristic(formula, beam[0])


# VND
def vnd(formula, vars_list, heuristic):
    current = {v: random.choice([True, False]) for v in vars_list}
    k = 1
    while k <= 3:
        improved = False
        for _ in range(k * 5):  
            neighbor = current.copy()
            flip_vars = random.sample(vars_list, k)
            for fv in flip_vars:
                neighbor[fv] = not neighbor[fv]
            if heuristic(formula, neighbor) > heuristic(formula, current):
                current = neighbor
                improved = True
                break
        if improved:
            k = 1
        else:
            k += 1
    return heuristic(formula, current)


# Run Program
def run_experiment():
    random.seed(0)
    settings = [(20, 5), (30, 8), (40, 10)]  
    for m, n in settings:
        print(f"\n==== 3-SAT problem with m={m}, n={n} ====")
        formula = make_3SAT(m, n)
        vars_list = [f"x{i+1}" for i in range(n)]

        for heuristic in [heuristic1, heuristic2]:
            hname = "Heuristic 1" if heuristic == heuristic1 else "Heuristic 2"
            print(f"\nUsing {hname}:")
            print("Hill-Climb:", hill_climb(formula, vars_list, heuristic))
            print("Beam (width=3):", beam_search(formula, vars_list, heuristic, 3))
            print("Beam (width=4):", beam_search(formula, vars_list, heuristic, 4))
            print("VND:", vnd(formula, vars_list, heuristic))

if __name__ == "__main__":
    run_experiment()
