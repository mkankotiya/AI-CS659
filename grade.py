"""
exp5_gaussian_hmm_demo.py
Demo of Gaussian HMM concept using synthetic returns.
No external modules required.
"""

import numpy as np

def synthetic_demo():
    np.random.seed(0)
    r1 = np.random.normal(0.0005, 0.005, 300)   # low-volatility regime
    r2 = np.random.normal(-0.0002, 0.02, 200)   # high-volatility regime
    returns = np.concatenate([r1, r2, r1])
    return returns

def detect_regimes(returns, threshold=0.01):
    """
    Simple heuristic to mimic HMM states:
    - abs(return) < threshold => low-volatility state
    - abs(return) >= threshold => high-volatility state
    """
    states = np.where(np.abs(returns) < threshold, 0, 1)
    return states

if __name__ == "__main__":
    returns = synthetic_demo()
    states = detect_regimes(returns)

    print("First 20 returns:")
    print(returns[:20])
    print("\nCorresponding detected states (0=low, 1=high):")
    print(states[:20])

    # Optional: simple text-based plot
    print("\nRegime visualization:")
    for i in range(50):  # first 50 returns
        marker = "*" if states[i] == 1 else "."
        print(marker, end="")
    print()
