"""
Monte Carlo — First-Visit Control
==================================
Model-free algorithm that learns Q(s,a) from complete sampled episodes.

1. Generate episode under current ε-greedy policy.
2. Walk backwards, compute discounted return G.
3. For each first-visited (s,a) pair, update:
       Q(s,a) ← average of all observed returns for (s,a)
4. Improve policy to be ε-greedy w.r.t. Q.

No bootstrapping — uses actual returns, so needs full episodes.
"""

import random
import numpy as np

from environment import (
    ROWS, COLS, START, GOAL, ACTIONS, GAMMA,
    get_valid_states, step,
)


def _generate_episode(policy_probs, max_steps=200):
    """
    Roll out one episode from START under a stochastic policy.

    Parameters
    ----------
    policy_probs : dict {state: list[float]} – π(a|s) for each state
    max_steps    : int – truncation limit

    Returns
    -------
    episode : list[(state, action_index, reward)]
    """
    episode = []
    s = START

    for _ in range(max_steps):
        probs = policy_probs.get(s, [0.25] * 4)
        a_idx = random.choices(range(4), weights=probs, k=1)[0]
        ns, r, done = step(s, ACTIONS[a_idx])
        episode.append((s, a_idx, r))
        s = ns
        if done:
            break

    return episode


def monte_carlo(n_episodes=3000, epsilon=0.2):
    """
    First-visit Monte Carlo control with ε-greedy exploration.

    Parameters
    ----------
    n_episodes : int   – total training episodes
    epsilon    : float – exploration rate

    Returns
    -------
    V       : np.ndarray (ROWS, COLS) – max Q per state
    policy  : dict {state: action_str}
    history : list[(episode_num, Q_snapshot)]
    """
    Q = {s: np.zeros(4) for s in get_valid_states()}
    returns_sum = {s: np.zeros(4) for s in get_valid_states()}
    returns_count = {s: np.zeros(4) for s in get_valid_states()}

    def _make_policy():
        """Build ε-greedy policy from current Q."""
        pp = {}
        for s in get_valid_states():
            if s == GOAL:
                pp[s] = [0.25] * 4
                continue
            best = int(np.argmax(Q[s]))
            probs = [epsilon / 4] * 4
            probs[best] += 1 - epsilon
            pp[s] = probs
        return pp

    policy_probs = _make_policy()
    history = []
    snapshot_interval = max(1, n_episodes // 30)

    for ep in range(n_episodes):
        episode = _generate_episode(policy_probs)

        # ── Backward pass: compute returns ──
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a_idx, r = episode[t]
            G = GAMMA * G + r

            # First-visit check
            if (s, a_idx) not in visited:
                visited.add((s, a_idx))
                returns_sum[s][a_idx] += G
                returns_count[s][a_idx] += 1
                Q[s][a_idx] = returns_sum[s][a_idx] / returns_count[s][a_idx]

        # ── Policy improvement ──
        policy_probs = _make_policy()

        if (ep + 1) % snapshot_interval == 0:
            history.append((ep + 1, {s: Q[s].copy() for s in Q}))

    # ── Derive final greedy policy ──
    policy = {}
    for s in get_valid_states():
        if s == GOAL:
            policy[s] = "★"
        else:
            policy[s] = ACTIONS[int(np.argmax(Q[s]))]

    V = np.zeros((ROWS, COLS))
    for s in get_valid_states():
        V[s] = np.max(Q[s])

    return V, policy, history
