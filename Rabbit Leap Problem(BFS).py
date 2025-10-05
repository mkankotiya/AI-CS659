# Rabbit Leap Problem using Breadth-First Search (BFS)

# Fundamental Rules of Rabbit Leap Problem
# -------------------------
# 1. EW -> WE       (E and W swap when facing each other)
# 2. EE -> E_E      (E can move one step right)
# 3. _W -> W_       (W moves left into empty space)
# 4. E_ -> _E       (E moves right into empty space)
# 5. EW_ -> _WE     (E jumps over W to right side)
# Rule 3 is applied first, so the first move from EEE_WWW is -> EEEW_WW

from collections import deque

# generate all possible next states
def next_moves(curr_state):
    moves = []
    gap = curr_state.index('_')   # find the empty stone position
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


# BFS Implementation
def bfs_rabbit(start_state, goal_state):
    queue = deque([[start_state]]) 
    seen = set()

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current in seen:
            continue
        seen.add(current)

        if current == goal_state:
            return path

        for move in next_moves(current):
            new_path = list(path)
            new_path.append(move)
            queue.append(new_path)

    return None



# start and goal states
start_state = ('E', 'E', 'E', '_', 'W', 'W', 'W')
goal_state  = ('W', 'W', 'W', '_', 'E', 'E', 'E')


solution_path = bfs_rabbit(start_state, goal_state)

# Print the result
if solution_path:
    print("\n✅ Solution Found using BFS!\n")
    for step_no, state in enumerate(solution_path):
        print(f"Step {step_no}: {state}")
    print(f"\nTotal Moves: {len(solution_path) - 1}")
else:
    print("❌ No Solution Found")

