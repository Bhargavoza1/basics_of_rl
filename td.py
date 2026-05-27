"""
TD — Q-Learning (Off-Policy TD Control)
=========================================
Watkins (1989).  Model-free algorithm that learns Q(s,a) using
**bootstrapping** — it updates after every single step, not at the
end of an episode.

Update rule (after each step):
    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

The key insight:  instead of waiting for a full episode to compute
the real return G_t (like Monte Carlo), TD *estimates* the return
using the current Q values of the next state.  The term

    r + γ · max_a' Q(s', a')

is called the **TD target**, and the difference between target and
current estimate is the **TD error** (δ).

Q-Learning is "off-policy" because it always uses max over actions
for the next state — it learns the optimal policy regardless of
which exploratory action it actually takes.
"""

import random
import numpy as np

from environment import (
    ROWS, COLS, START, GOAL, ACTIONS, GAMMA,
    get_valid_states, step,
)


def q_learning(n_episodes=2000, alpha=0.1, epsilon=0.2):
    """
    Off-policy TD control (Q-Learning).

    Parameters
    ----------
    n_episodes : int   – total training episodes
    alpha      : float – learning rate (step size)
    epsilon    : float – exploration rate for ε-greedy

    Returns
    -------
    V              : np.ndarray (ROWS, COLS) – max Q per state
    policy         : dict {state: action_str}
    history        : list[(episode_num, Q_snapshot)]
    reward_history : list[float] – total reward per episode
    """
    Q = {s: np.zeros(4) for s in get_valid_states()}
    history = []
    reward_history = []
    snapshot_interval = max(1, n_episodes // 30)

    for ep in range(n_episodes):
        s = START
        total_reward = 0.0

        for _ in range(200):
            # ── ε-greedy action selection ──
            if random.random() < epsilon:
                a_idx = random.randint(0, 3)
            else:
                a_idx = int(np.argmax(Q[s]))

            # ── Take action, observe outcome ──
            ns, r, done = step(s, ACTIONS[a_idx])
            total_reward += r

            # ── TD update (the core of Q-Learning) ──
            #
            #   TD target  = r + γ · max_a' Q(s', a')
            #   TD error   = TD target − Q(s, a)
            #   Q(s,a)    ← Q(s,a) + α · TD error
            #
            if done:
                td_target = r                             # no future after goal
            else:
                td_target = r + GAMMA * np.max(Q[ns])     # bootstrap from next state

            td_error = td_target - Q[s][a_idx]
            Q[s][a_idx] += alpha * td_error

            s = ns
            if done:
                break

        reward_history.append(total_reward)

        if (ep + 1) % snapshot_interval == 0:
            history.append((ep + 1, {s: Q[s].copy() for s in Q}))

    # ── Derive greedy policy ──
    policy = {}
    for s in get_valid_states():
        if s == GOAL:
            policy[s] = "★"
        else:
            policy[s] = ACTIONS[int(np.argmax(Q[s]))]

    V = np.zeros((ROWS, COLS))
    for s in get_valid_states():
        V[s] = np.max(Q[s])

    return V, policy, history, reward_history
