import numpy as np
from math import exp, factorial

# ============================================================
# PART 1: 4x3 GRIDWORLD VALUE ITERATION (IN-LAB PROBLEM)
# ============================================================

class GridworldMDP:
    """
    4x3 gridworld as in Sutton/Russell-Norvig:
    Coordinates: (x, y) with x=0..3, y=0..2
    Terminal states: +1 at (3, 2), -1 at (3, 1)
    Wall: (1, 1)
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Stochastic:
        - Intended direction: 0.8
        - Right-angles: 0.1 each
        - Hit wall/boundary -> stays in same state.
    """
    def __init__(self, step_reward=-0.04, gamma=1.0):
        self.width = 4
        self.height = 3
        self.gamma = gamma
        self.step_reward = step_reward

        self.terminal_states = {
            (3, 2): 1.0,   # +1
            (3, 1): -1.0   # -1
        }
        self.wall = (1, 1)

        # Actions: up, right, down, left
        self.actions = {
            0: (0, 1),   # Up
            1: (1, 0),   # Right
            2: (0, -1),  # Down
            3: (-1, 0)   # Left
        }

    def in_bounds(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) == self.wall:
            return False
        return True

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_states(self):
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) == self.wall:
                    continue
                yield (x, y)

    def transitions(self, state, action):
        """
        Returns list of (prob, next_state, reward)
        """
        if self.is_terminal(state):
            # For terminal states, no transitions (absorbing)
            return [(1.0, state, self.terminal_states[state])]

        x, y = state
        # Intended + orthogonal actions
        intended = action
        right = (action + 1) % 4
        left = (action - 1) % 4

        outcomes = [
            (0.8, intended),
            (0.1, right),
            (0.1, left)
        ]

        results = []
        for prob, a in outcomes:
            dx, dy = self.actions[a]
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                nx, ny = x, y  # stay in place
            next_state = (nx, ny)

            if next_state in self.terminal_states:
                reward = self.terminal_states[next_state]
            else:
                reward = self.step_reward

            results.append((prob, next_state, reward))
        return results


def value_iteration_gridworld(step_reward=-0.04, gamma=1.0, theta=1e-4, max_iter=1000):
    mdp = GridworldMDP(step_reward=step_reward, gamma=gamma)
    V = {s: 0.0 for s in mdp.get_states()}

    for it in range(max_iter):
        delta = 0.0
        new_V = V.copy()
        for s in mdp.get_states():
            if mdp.is_terminal(s):
                new_V[s] = mdp.terminal_states[s]
                continue

            # Bellman optimality
            action_values = []
            for a in mdp.actions.keys():
                val = 0.0
                for prob, ns, r in mdp.transitions(s, a):
                    val += prob * (r + mdp.gamma * V[ns])
                action_values.append(val)
            best = max(action_values)
            delta = max(delta, abs(best - V[s]))
            new_V[s] = best
        V = new_V

        if delta < theta:
            print(f"Value iteration converged in {it+1} iterations.")
            break

    # Derive greedy policy
    policy = {}
    for s in mdp.get_states():
        if mdp.is_terminal(s):
            policy[s] = None
            continue
        best_a = None
        best_val = -1e9
        for a in mdp.actions.keys():
            val = 0.0
            for prob, ns, r in mdp.transitions(s, a):
                val += prob * (r + mdp.gamma * V[ns])
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a

    return V, policy


def print_grid_values(V, title="Values"):
    print(f"\n{title}")
    # y from top (2) to bottom (0)
    for y in reversed(range(3)):
        row = []
        for x in range(4):
            if (x, y) == (1, 1):
                row.append("####")
            else:
                row.append(f"{V[(x, y)]:6.2f}")
        print(" ".join(row))


# ============================================================
# PART 2: GBIKE BICYCLE RENTAL (POLICY ITERATION)
# ============================================================

# --- Poisson helper ---
def poisson_pmf(lmbda, n):
    return exp(-lmbda) * (lmbda ** n) / factorial(n)

def precompute_poisson(lmbda, max_n=11):
    """
    Precompute Poisson probabilities from 0..max_n, 
    with a tail probability folded into max_n.
    """
    probs = [poisson_pmf(lmbda, n) for n in range(max_n)]
    tail = 1.0 - sum(probs)
    probs.append(tail)
    return probs


class GbikeEnv:
    """
    Environment for the Gbike bicycle rental problem (Jack's car rental style).
    State: (i, j) where i = bikes at location 1, j = bikes at location 2.
    Action: number of bikes moved from loc1 -> loc2 overnight (negative => move 2->1).
    """

    def __init__(
        self,
        max_bikes=20,
        max_move=5,
        rent_reward=10,
        move_cost=2,
        lambda_req1=3,
        lambda_req2=4,
        lambda_ret1=3,
        lambda_ret2=2,
        gamma=0.9,
        free_move_1_to_2=False,
        parking_penalty=False
    ):
        self.max_bikes = max_bikes
        self.max_move = max_move
        self.rent_reward = rent_reward
        self.move_cost = move_cost
        self.gamma = gamma

        self.free_move_1_to_2 = free_move_1_to_2
        self.parking_penalty = parking_penalty

        self.poisson_max = 11
        self.req1_probs = precompute_poisson(lambda_req1, self.poisson_max)
        self.req2_probs = precompute_poisson(lambda_req2, self.poisson_max)
        self.ret1_probs = precompute_poisson(lambda_ret1, self.poisson_max)
        self.ret2_probs = precompute_poisson(lambda_ret2, self.poisson_max)

        # all states
        self.states = [(i, j) for i in range(max_bikes + 1) for j in range(max_bikes + 1)]

    def valid_actions(self, state):
        i, j = state
        actions = []
        # action = bikes moved from 1 to 2 (negative => 2 to 1)
        for a in range(-self.max_move, self.max_move + 1):
            # free move logic: one bike from 1->2 is free; above that cost applies
            # but feasibility of the move is purely based on capacity/availability
            if a > 0 and i >= a and j + a <= self.max_bikes:
                actions.append(a)
            elif a < 0 and j >= -a and i - a <= self.max_bikes:
                actions.append(a)
            elif a == 0:
                actions.append(a)
        return actions

    def expected_return(self, state, action, V):
        """
        Compute expected return for given state, action under current value function V.
        Following Sutton & Barto style enumeration of Poisson requests and returns.
        """
        i, j = state

        # Apply action: move bikes overnight
        bikes1 = i - action
        bikes2 = j + action
        move_cost = self.move_cost * abs(action)

        # Modification: one free bike from 1->2
        if self.free_move_1_to_2 and action > 0:
            # One bike free, rest charged
            move_cost = self.move_cost * max(0, abs(action) - 1)

        if bikes1 < 0 or bikes1 > self.max_bikes or bikes2 < 0 or bikes2 > self.max_bikes:
            # invalid move, we can treat as very bad return
            return -1e9

        # Parking penalty (after returns)
        # We'll add this later when computing final next state.

        expected_ret = -move_cost  # immediate cost for moving bikes

        # Enumerate over rental requests and returns
        for req1 in range(self.poisson_max + 1):
            p_req1 = self.req1_probs[req1]
            real_req1 = min(bikes1, req1)
            reward1 = real_req1 * self.rent_reward
            bikes1_after_rent = bikes1 - real_req1

            for req2 in range(self.poisson_max + 1):
                p_req2 = self.req2_probs[req2]
                real_req2 = min(bikes2, req2)
                reward2 = real_req2 * self.rent_reward
                bikes2_after_rent = bikes2 - real_req2

                prob_req = p_req1 * p_req2
                reward_rent = reward1 + reward2

                # Returns
                for ret1 in range(self.poisson_max + 1):
                    p_ret1 = self.ret1_probs[ret1]
                    bikes1_end = min(bikes1_after_rent + ret1, self.max_bikes)

                    for ret2 in range(self.poisson_max + 1):
                        p_ret2 = self.ret2_probs[ret2]
                        bikes2_end = min(bikes2_after_rent + ret2, self.max_bikes)

                        prob = prob_req * p_ret1 * p_ret2
                        next_state = (bikes1_end, bikes2_end)
                        extra_penalty = 0

                        # Parking penalty: if > 10 bikes at a location overnight
                        if self.parking_penalty:
                            if bikes1_end > 10:
                                extra_penalty -= 4
                            if bikes2_end > 10:
                                extra_penalty -= 4

                        total_reward = reward_rent + extra_penalty

                        expected_ret += prob * (total_reward + self.gamma * V[next_state])

        return expected_ret


def policy_evaluation(env: GbikeEnv, policy, theta=1e-2):
    V = {s: 0.0 for s in env.states}
    while True:
        delta = 0.0
        for s in env.states:
            v = V[s]
            a = policy[s]
            V[s] = env.expected_return(s, a, V)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(env: GbikeEnv, V, policy):
    policy_stable = True
    for s in env.states:
        old_action = policy[s]
        actions = env.valid_actions(s)
        best_a = None
        best_val = -1e9
        for a in actions:
            val = env.expected_return(s, a, V)
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a
        if best_a != old_action:
            policy_stable = False
    return policy_stable, policy


def policy_iteration(env: GbikeEnv, theta=1e-2):
    # initial policy: do nothing
    policy = {s: 0 for s in env.states}
    iteration = 0
    while True:
        iteration += 1
        print(f"Policy iteration step {iteration}")
        V = policy_evaluation(env, policy, theta=theta)
        stable, policy = policy_improvement(env, V, policy)
        if stable:
            print("Policy stable, stopping.")
            break
    return policy, V


def print_policy(env: GbikeEnv, policy):
    print("\nOptimal policy (bikes moved from loc1 -> loc2):")
    for i in range(env.max_bikes, -1, -1):
        row = []
        for j in range(env.max_bikes + 1):
            row.append(f"{policy[(i, j)]:2d}")
        print(f"{i:2d} | " + " ".join(row))
    print("    j = 0..20 across columns, i = 20..0 down rows")


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":
    # ---------------- Gridworld value iteration --------------
    rewards_to_try = [-0.04, -2.0, 0.1, 0.02, 1.0]
    for r in rewards_to_try:
        V, policy = value_iteration_gridworld(step_reward=r, gamma=1.0)
        print_grid_values(V, title=f"Gridworld state values for r(s) = {r}")

    # ---------------- Gbike base problem ---------------------
    print("\n=== Gbike base problem (original Jack's car rental style) ===")
    base_env = GbikeEnv(
        max_bikes=20,
        max_move=5,
        rent_reward=10,
        move_cost=2,
        lambda_req1=3,
        lambda_req2=4,
        lambda_ret1=3,
        lambda_ret2=2,
        gamma=0.9,
        free_move_1_to_2=False,
        parking_penalty=False
    )
    base_policy, base_V = policy_iteration(base_env, theta=1.0)
    # You can print a subset or whole policy (it is big)
    # print_policy(base_env, base_policy)

    # ---------------- Gbike modified problem -----------------
    print("\n=== Gbike modified problem (free move + parking penalty) ===")
    mod_env = GbikeEnv(
        max_bikes=20,
        max_move=5,
        rent_reward=10,
        move_cost=2,
        lambda_req1=3,
        lambda_req2=4,
        lambda_ret1=3,
        lambda_ret2=2,
        gamma=0.9,
        free_move_1_to_2=True,    # 1 free bike from loc1->loc2
        parking_penalty=True      # extra cost if > 10 bikes
    )
    mod_policy, mod_V = policy_iteration(mod_env, theta=1.0)
    # print_policy(mod_env, mod_policy)
