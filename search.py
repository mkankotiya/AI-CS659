import numpy as np
import random
from collections import defaultdict

# ============================================================
# PART 1: BANDITS
#   (2) Binary bandit with epsilon-greedy
#   (3) Non-stationary 10-armed bandit
#   (4) Modified epsilon-greedy for non-stationary bandit
# ============================================================

# -------------------------------
# (2) Binary Bandit (stationary)
# -------------------------------

class BinaryBandit:
    """Two-armed bandit with Bernoulli rewards."""
    def __init__(self, p1=0.7, p2=0.4):
        self.p = [p1, p2]   # success probabilities for actions 0 and 1

    def pull(self, action: int) -> int:
        """Return 1 (success) or 0 (failure)."""
        return 1 if np.random.rand() < self.p[action] else 0


def epsilon_greedy_bandit(bandit: BinaryBandit, epsilon=0.1, steps=1000):
    """
    Standard epsilon-greedy algorithm for a stationary 2-armed bandit.
    Incremental sample-average update for Q-values.
    """
    Q = [0.0, 0.0]   # estimated values for actions 0 and 1
    N = [0, 0]       # number of times each action is selected
    rewards = []

    for t in range(steps):
        # Exploration vs exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)   # explore
        else:
            action = int(np.argmax(Q))         # exploit

        r = bandit.pull(action)
        N[action] += 1
        # Incremental mean update
        Q[action] += (r - Q[action]) / N[action]

        rewards.append(r)

    return Q, rewards


# --------------------------------------------
# (3) Non-stationary 10-armed Random-Walk Bandit
# --------------------------------------------

class NonStationaryBandit:
    """
    10-armed bandit where each arm’s mean reward follows a random walk.
    Reward ~ N(mean[action], 1), and means drift each time step.
    """
    def __init__(self, arms=10, walk_std=0.01):
        self.arms = arms
        self.walk_std = walk_std
        self.means = np.zeros(arms)  # start all means equal

    def pull(self, action: int) -> float:
        reward = np.random.normal(self.means[action], 1.0)
        # Random walk for all arms
        self.means += np.random.normal(0.0, self.walk_std, self.arms)
        return reward


# --------------------------------------------------------
# (4) Modified epsilon-greedy for Non-stationary Bandit
# --------------------------------------------------------

def modified_epsilon_greedy(bandit: NonStationaryBandit,
                            epsilon=0.1,
                            alpha=0.1,
                            steps=10000):
    """
    Epsilon-greedy with constant step-size alpha for non-stationary rewards.
    Q_{t+1}(A) = Q_t(A) + alpha * (R_t - Q_t(A))
    """
    arms = bandit.arms
    Q = np.zeros(arms)
    rewards = []
    chosen_actions = []

    for t in range(steps):
        # Exploration vs exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(arms)   # explore
        else:
            action = int(np.argmax(Q))         # exploit

        r = bandit.pull(action)

        # Constant step-size update
        Q[action] = Q[action] + alpha * (r - Q[action])

        rewards.append(r)
        chosen_actions.append(action)

    return Q, rewards, chosen_actions


# ============================================================
# PART 2: MENACE (MATCHBOX TIC-TAC-TOE ENGINE)
# ============================================================

# -------------------------------
# Tic-Tac-Toe Environment
# -------------------------------

class TicTacToe:
    def __init__(self):
        # Board is a list of 9 cells: 'X', 'O', or ' '
        self.board = [' '] * 9
        self.current_player = 'X'  # MENACE will play 'X' by default

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def available_moves(self):
        return [i for i, c in enumerate(self.board) if c == ' ']

    def make_move(self, pos: int):
        if self.board[pos] != ' ':
            raise ValueError("Invalid move")
        self.board[pos] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def winner(self):
        b = self.board
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
            (0, 4, 8), (2, 4, 6)               # diagonals
        ]
        for i, j, k in lines:
            if b[i] == b[j] == b[k] != ' ':
                return b[i]
        if ' ' not in b:
            return 'D'  # Draw
        return None     # Game not over

    def game_over(self) -> bool:
        return self.winner() is not None

    def board_key(self) -> str:
        """Return a simple representation of the board for MENACE."""
        return ''.join(self.board)

    def print_board(self):
        b = self.board
        print(f"{b[0]}|{b[1]}|{b[2]}")
        print("-+-+-")
        print(f"{b[3]}|{b[4]}|{b[5]}")
        print("-+-+-")
        print(f"{b[6]}|{b[7]}|{b[8]}")
        print()


# -------------------------------
# MENACE Agent
# -------------------------------

class MenaceAgent:
    """
    MENACE-like agent using matchboxes:
    - For each state (board config), we store a dict: action -> bead count.
    - Action chosen randomly with probability proportional to bead count.
    - Learning: reward/punish beads after game ends.
    """

    def __init__(self, player='X', initial_beads=3, min_beads=1):
        self.player = player          # 'X' or 'O'
        self.initial_beads = initial_beads
        self.min_beads = min_beads

        # state_key -> {action_index: bead_count}
        self.matchboxes = defaultdict(dict)

        # list of (state_key, action) during one game
        self.episode_moves = []

    def _init_matchbox_if_needed(self, state_key, available_moves):
        if not self.matchboxes[state_key]:
            for move in available_moves:
                self.matchboxes[state_key][move] = self.initial_beads

    def select_action(self, env: TicTacToe) -> int:
        """
        Choose move for current state based on bead counts.
        """
        state_key = env.board_key()
        moves = env.available_moves()

        # initialize if needed
        self._init_matchbox_if_needed(state_key, moves)

        beads = self.matchboxes[state_key]
        actions = list(beads.keys())
        weights = [beads[a] for a in actions]

        # stochastic choice
        action = random.choices(actions, weights=weights, k=1)[0]

        # store trajectory for learning
        self.episode_moves.append((state_key, action))
        return action

    def learn_from_game(self, result: str):
        """
        Apply MENACE-style reinforcement:
            win  -> +3 beads
            draw -> +1 bead
            loss -> -1 bead (but not below min_beads)
        result is 'win', 'draw', or 'loss' from MENACE's perspective.
        """
        if result == 'win':
            delta = 3
        elif result == 'draw':
            delta = 1
        else:  # 'loss'
            delta = -1

        for state_key, action in self.episode_moves:
            beads = self.matchboxes[state_key]
            beads[action] = max(self.min_beads, beads[action] + delta)

        # clear episode
        self.episode_moves = []


# -------------------------------
# Opponent Agent (Random)
# -------------------------------

class RandomAgent:
    def __init__(self, player='O'):
        self.player = player

    def select_action(self, env: TicTacToe) -> int:
        return random.choice(env.available_moves())


# -------------------------------
# Play one game MENACE vs Opponent
# -------------------------------

def play_game(env: TicTacToe,
              menace: MenaceAgent,
              opponent,
              verbose=False) -> str:
    """
    Play a single game. Returns 'win', 'draw', or 'loss' for MENACE.
    """
    env.reset()
    menace.episode_moves = []

    while not env.game_over():
        if env.current_player == menace.player:
            move = menace.select_action(env)
        else:
            move = opponent.select_action(env)

        env.make_move(move)

        if verbose:
            env.print_board()

    w = env.winner()
    if w == menace.player:
        result = 'win'
    elif w == 'D':
        result = 'draw'
    else:
        result = 'loss'

    menace.learn_from_game(result)
    return result


# -------------------------------
# Training & Testing MENACE
# -------------------------------

def train_menace(num_games=5000, verbose_every=1000):
    env = TicTacToe()
    menace = MenaceAgent(player='X', initial_beads=3, min_beads=1)
    opponent = RandomAgent(player='O')

    stats = {'win': 0, 'draw': 0, 'loss': 0}

    for i in range(1, num_games + 1):
        result = play_game(env, menace, opponent, verbose=False)
        stats[result] += 1

        if verbose_every is not None and i % verbose_every == 0:
            total = i
            print(f"[Train] After {total} games:")
            print(f"  Wins : {stats['win']} ({stats['win']/total:.2f})")
            print(f"  Draws: {stats['draw']} ({stats['draw']/total:.2f})")
            print(f"  Loss : {stats['loss']} ({stats['loss']/total:.2f})")
            print("-" * 35)

    return menace, stats


def test_menace(menace: MenaceAgent, num_games=1000):
    env = TicTacToe()
    opponent = RandomAgent(player='O')
    stats = {'win': 0, 'draw': 0, 'loss': 0}

    for _ in range(num_games):
        result = play_game(env, menace, opponent, verbose=False)
        stats[result] += 1

    total = num_games
    print("[Test] MENACE vs Random Opponent")
    print(f"  Wins : {stats['win']} ({stats['win']/total:.2f})")
    print(f"  Draws: {stats['draw']} ({stats['draw']/total:.2f})")
    print(f"  Loss : {stats['loss']} ({stats['loss']/total:.2f})")
    print("-" * 35)


# ============================================================
# MAIN: small demo runs
# ============================================================

if __name__ == "__main__":

    # ------------------ Binary Bandit Demo ------------------
    print("=== Binary Bandit with Epsilon-Greedy ===")
    banditA = BinaryBandit(0.8, 0.3)
    Q_bin, rewards_bin = epsilon_greedy_bandit(banditA, epsilon=0.1, steps=1000)
    print("Final Q-values:", Q_bin)
    print("Average Reward:", np.mean(rewards_bin))
    print()

    # ------------- Non-stationary Bandit Demo --------------
    print("=== Non-stationary 10-armed Bandit (Modified Epsilon-Greedy) ===")
    nonstat_bandit = NonStationaryBandit(arms=10, walk_std=0.01)
    Q_ns, rewards_ns, actions_ns = modified_epsilon_greedy(
        nonstat_bandit, epsilon=0.1, alpha=0.1, steps=10000
    )
    print("Final Q-values (first 5):", Q_ns[:5])
    print("Average Reward:", np.mean(rewards_ns))
    print("Most chosen action:", np.argmax(np.bincount(actions_ns)))
    print()

    # -------------------- MENACE Training -------------------
    print("=== Training MENACE (Tic-Tac-Toe) ===")
    menace_agent, train_stats = train_menace(num_games=5000, verbose_every=1000)

    # -------------------- MENACE Testing --------------------
    test_menace(menace_agent, num_games=1000)

    # If you want human vs MENACE, you can add interactive code here.
