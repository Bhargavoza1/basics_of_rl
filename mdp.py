"""
MDP — Value Iteration
=====================
Synchronous Bellman optimality backup.
Model-based: uses full knowledge of transitions P(s'|s,a).

    V(s) ← max_a  Σ P(s'|s,a) [R(s,a,s') + γ V(s')]

Since transitions are deterministic here, the sum collapses to a
single next-state lookup.

Returns the converged value function V, the greedy policy, and a
snapshot history for animation.
"""

import numpy as np

from environment import (
    ROWS, COLS, GOAL, ACTIONS, GAMMA,
    get_valid_states, step,
)


def value_iteration(theta=1e-4, max_iters=500):
    """
    Run value iteration until convergence.

    Parameters
    ----------
    theta     : float – convergence threshold on max |ΔV|
    max_iters : int   – safety cap on sweeps

    Returns
    -------
    V       : np.ndarray (ROWS, COLS) – optimal state values
    policy  : dict {(r,c): action_str}
    history : list[np.ndarray] – V snapshot after each sweep
    """
    V = np.zeros((ROWS, COLS))
    history = []

    for iteration in range(max_iters):
        delta = 0.0
        new_V = V.copy()

        for s in get_valid_states():
            if s == GOAL:
                continue

            # Bellman optimality backup
            action_values = []
            for a in ACTIONS:
                ns, r, _ = step(s, a)
                action_values.append(r + GAMMA * V[ns])

            best = max(action_values)
            delta = max(delta, abs(best - V[s]))
            new_V[s] = best

        V = new_V
        history.append(V.copy())

        if delta < theta:
            break

    # ── Extract greedy policy from converged V ──
    policy = {}
    for s in get_valid_states():
        if s == GOAL:
            policy[s] = "★"
            continue

        best_action = ACTIONS[0]
        best_value = -1e9
        for a in ACTIONS:
            ns, r, _ = step(s, a)
            v = r + GAMMA * V[ns]
            if v > best_value:
                best_value = v
                best_action = a
        policy[s] = best_action

    return V, policy, history
