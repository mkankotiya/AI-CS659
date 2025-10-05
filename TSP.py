"""
exp4_simulated_annealing_jigsaw.py
A simplified jigsaw-like puzzle solver using simulated annealing.
"""
import random, math

def make_patches(n):
    patches = []
    for i in range(n*n):
        row, col = divmod(i, n)
        right = (row*100 + col + 1) if col < n-1 else None
        bottom = ((row+1)*100 + col) if row < n-1 else None
        patches.append({'id': i, 'right': right, 'bottom': bottom})
    return patches

def score_arrangement(arr, n):
    total = 0
    for idx, p in enumerate(arr):
        row, col = divmod(idx, n)
        if col < n-1:
            if p['right'] != None and arr[idx+1]['id'] != (p['right'] % n): total += 1
        if row < n-1:
            if p['bottom'] != None and arr[idx+n]['id'] != (p['bottom'] % n): total += 1
    return total

def simulated_annealing(patches, n, T0=10.0, alpha=0.995, steps=20000):
    curr = patches[:]; random.shuffle(curr)
    curr_score = score_arrangement(curr, n)
    best, best_score, T = curr[:], curr_score, T0
    for step in range(steps):
        i,j = random.sample(range(len(curr)), 2)
        curr[i], curr[j] = curr[j], curr[i]
        sc = score_arrangement(curr, n)
        delta = sc - curr_score
        if delta < 0 or random.random() < math.exp(-delta/T):
            curr_score = sc
            if sc < best_score: best, best_score = curr[:], sc
        else: curr[i], curr[j] = curr[j], curr[i]
        T *= alpha
    return best, best_score

if __name__ == '__main__':
    n = 3
    patches = make_patches(n)
    best, score = simulated_annealing(patches, n)
    print("Best arrangement IDs:", [p['id'] for p in best], "with score", score)
