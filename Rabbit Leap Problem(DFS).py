# Rabbit Leap Problem using Depth-First Search (DFS)

# Fundamental Rules of Rabbit Leap Problem
# -------------------------
# 1. EW -> WE       (E and W swap when facing each other)
# 2. EE -> E_E      (E can move one step right)
# 3. _W -> W_       (W moves left into empty space)
# 4. E_ -> _E       (E moves right into empty space)
# 5. EW_ -> _WE     (E jumps over W to right side)
# Rule 3 is applied first, so the first move from EEE_WWW is -> EEEW_WW

# generate all possible next states
def next_moves(curr_state):
    moves = []
    gap = curr_state.index('_')  
    total = len(curr_state)

    # 3: _W -> W_  (W moves left into empty)
    if gap < total - 1 and curr_state[gap + 1] == 'W':
        s = list(curr_state)
        s[gap], s[gap + 1] = s[gap + 1], s[gap]
        moves.append(tuple(s))

    # 1: EW -> WE  (W jumps left over E)
    if gap < total - 2 and curr_state[gap + 2] == 'W' and curr_state[gap + 1] == 'E':
        s = list(curr_state)
        s[gap], s[gap + 2] = s[gap + 2], s[gap]
        moves.append(tuple(s))

    # 4: E_ -> _E  (E moves right into empty)
    if gap > 0 and curr_state[gap - 1] == 'E':
        s = list(curr_state)
        s[gap], s[gap - 1] = s[gap - 1], s[gap]
        moves.append(tuple(s))

    # 5: EW_ -> _WE  (E jumps right over W)
    if gap > 1 and curr_state[gap - 2] == 'E' and curr_state[gap - 1] == 'W':
        s = list(curr_state)
        s[gap], s[gap - 2] = s[gap - 2], s[gap]
        moves.append(tuple(s))

    # 2: EE -> E_E  (E moves one step right)
    if gap > 1 and curr_state[gap - 1] == 'E' and curr_state[gap - 2] == 'E':
        s = list(curr_state)
        s[gap - 1], s[gap] = s[gap], s[gap - 1]
        moves.append(tuple(s))

    return moves


# DFS implementation
def dfs_rabbit(current, goal, visited, path):
    visited.add(current)
    path.append(current)

    if current == goal:
        return True

    # all next possible moves (Depth First)
    for nxt in next_moves(current):
        if nxt not in visited:
            found = dfs_rabbit(nxt, goal, visited, path)
            if found:
                return True

    # remove last state if dead-end
    path.pop()
    return False


# start and goal states
start_state = ('E', 'E', 'E', '_', 'W', 'W', 'W')
goal_state  = ('W', 'W', 'W', '_', 'E', 'E', 'E')

visited = set()
solution_path = []

found = dfs_rabbit(start_state, goal_state, visited, solution_path)


# Print the result
if found:
    print("\n✅ Solution Found using DFS!\n")
    for step_no, state in enumerate(solution_path):
        print(f"Step {step_no}: {state}")
    print(f"\nTotal Moves: {len(solution_path) - 1}")
else:
    print("❌ No Solution Found")
