"""Generation-0 seed heuristic.

Random vertices anywhere, random gray, keep the best of K random tries.
Deliberately weak — it stops finding improving moves late in the budget
(visible slot starvation), which is exactly the floor the model must improve.
"""

BASELINE_SOURCE = '''\
# Baseline: uniformly random vertices and random gray; keep the best of K tries.
def propose_triangle(target, canvas, t, n_total, evaluate, rng):
    H, W = target.shape
    best = None
    for _ in range(K):
        pts = rng.integers(0, W, size=(3, 2))
        r = evaluate(pts, rng.random())
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    return best
'''
