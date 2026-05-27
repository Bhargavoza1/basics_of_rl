"""
REINFORCE — Policy Gradient (no baseline)
==========================================
Williams (1992).  Pure policy-based method — no value function.

Policy:  tabular softmax
    π(a|s) = exp(θ_{s,a}) / Σ_a' exp(θ_{s,a'})

Update rule (after each episode):
    θ ← θ + α · γ^t · G_t · ∇_θ log π(a_t | s_t)

For a softmax policy the score function simplifies to:
    ∇_θ log π(a|s) = e_a − π(·|s)

where e_a is the one-hot vector for the selected action.
"""

import random
import math
import numpy as np

from environment import (
    ROWS, COLS, START, GOAL, ACTIONS, GAMMA,
    get_valid_states, step,
)


class SoftmaxPolicy:
    """
    Tabular softmax policy parameterised by θ ∈ R^{|S| × |A|}.
    """

    def __init__(self):
        self.theta = {s: np.zeros(4) for s in get_valid_states()}

    def probs(self, s):
        """Return π(·|s) as a numpy array of length 4."""
        logits = self.theta[s] - np.max(self.theta[s])   # numerical stability
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def sample(self, s):
        """Sample an action index from π(·|s)."""
        p = self.probs(s)
        return random.choices(range(4), weights=p, k=1)[0]

    def log_prob(self, s, a_idx):
        """Return log π(a|s)."""
        p = self.probs(s)
        return math.log(p[a_idx] + 1e-12)


def reinforce(n_episodes=4000, lr=0.01):
    """
    REINFORCE without baseline.

    Parameters
    ----------
    n_episodes : int   – training episodes
    lr         : float – learning rate α

    Returns
    -------
    V              : np.ndarray (ROWS, COLS) – max π per state (for colour)
    policy         : dict {state: action_str}
    history        : list[(ep_num, theta_snapshot)]
    reward_history : list[float] – total reward per episode
    """
    pol = SoftmaxPolicy()
    history = []
    reward_history = []
    snapshot_interval = max(1, n_episodes // 30)

    for ep in range(n_episodes):

        # ── Generate episode ──
        trajectory = []
        s = START
        for _ in range(200):
            a_idx = pol.sample(s)
            ns, r, done = step(s, ACTIONS[a_idx])
            trajectory.append((s, a_idx, r))
            s = ns
            if done:
                break

        # ── Compute discounted returns (backward) ──
        G = 0.0
        returns = []
        for _, _, r in reversed(trajectory):
            G = GAMMA * G + r
            returns.insert(0, G)

        total_reward = sum(r for _, _, r in trajectory)
        reward_history.append(total_reward)

        # ── Policy gradient update ──
        for t, (s, a_idx, _) in enumerate(trajectory):
            p = pol.probs(s)

            # Score function for softmax: ∇ log π(a|s) = e_a − π
            onehot = np.zeros(4)
            onehot[a_idx] = 1.0
            score = onehot - p

            pol.theta[s] += lr * (GAMMA ** t) * returns[t] * score

        if (ep + 1) % snapshot_interval == 0:
            history.append(
                (ep + 1, {s: pol.theta[s].copy() for s in pol.theta})
            )

    # ── Derive greedy policy ──
    policy = {}
    V = np.zeros((ROWS, COLS))
    for s in get_valid_states():
        if s == GOAL:
            policy[s] = "★"
        else:
            probs = pol.probs(s)
            policy[s] = ACTIONS[int(np.argmax(probs))]
            V[s] = np.max(probs)   # probability as intensity for colouring

    return V, policy, history, reward_history
