"""
Actor-Critic (One-Step, Tabular)
=================================
Combines the two branches of RL:

    Actor  = policy π(a|s), parameterised by θ  (from REINFORCE)
    Critic = state-value V(s), learned with TD   (from TD Q-Learning)

The critic evaluates how good a state is.  The actor uses the
critic's TD error as an advantage signal to update the policy.

Update rules (after EVERY step, not end of episode):

    δ       = r + γ · V(s') − V(s)          ← TD error = advantage estimate
    V(s)   ← V(s)   + α_c · δ              ← critic update (TD learning)
    θ_{s,·}← θ_{s,·}+ α_a · γ^t · δ · ∇ log π(a|s)   ← actor update (policy gradient)

Key insight: REINFORCE used the full return G_t to judge actions,
which meant waiting until episode end and suffering high variance.
Actor-Critic replaces G_t with the TD error δ — a one-step
advantage estimate — giving immediate feedback with much lower
variance.

δ > 0 → action was BETTER than expected  → increase its probability
δ < 0 → action was WORSE  than expected  → decrease its probability
δ ≈ 0 → action was about as expected     → barely change anything
"""

import random
import math
import numpy as np

from environment import (
    ROWS, COLS, START, GOAL, ACTIONS, GAMMA,
    get_valid_states, step,
)


class ActorPolicy:
    """
    Tabular softmax policy (the Actor).
    Identical to REINFORCE's policy — the difference is in how it's trained.
    """

    def __init__(self):
        self.theta = {s: np.zeros(4) for s in get_valid_states()}

    def probs(self, s):
        """Return π(·|s) as a numpy array of length 4."""
        logits = self.theta[s] - np.max(self.theta[s])
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def sample(self, s):
        """Sample an action index from π(·|s)."""
        p = self.probs(s)
        return random.choices(range(4), weights=p, k=1)[0]


def actor_critic(n_episodes=2000, lr_actor=0.01, lr_critic=0.1):
    """
    One-step Actor-Critic.

    Parameters
    ----------
    n_episodes : int   – training episodes
    lr_actor   : float – learning rate for the actor (policy)
    lr_critic  : float – learning rate for the critic (value)

    Returns
    -------
    V              : np.ndarray (ROWS, COLS) – learned state values
    policy         : dict {state: action_str}
    history        : list[(ep_num, theta_snapshot, V_snapshot)]
    reward_history : list[float] – total reward per episode
    """
    actor = ActorPolicy()
    V = {s: 0.0 for s in get_valid_states()}  # Critic's value table
    history = []
    reward_history = []
    snapshot_interval = max(1, n_episodes // 30)

    for ep in range(n_episodes):
        s = START
        total_reward = 0.0
        t = 0  # time step within episode

        for _ in range(200):
            # ── Actor selects action ──
            a_idx = actor.sample(s)
            ns, r, done = step(s, ACTIONS[a_idx])
            total_reward += r

            # ── Critic computes TD error (= advantage estimate) ──
            #
            #   δ = r + γ · V(s') − V(s)
            #
            #   If s' is terminal (goal), V(s') = 0
            #
            if done:
                td_error = r - V[s]
            else:
                td_error = r + GAMMA * V[ns] - V[s]

            # ── Critic update: V(s) ← V(s) + α_c · δ ──
            V[s] += lr_critic * td_error

            # ── Actor update: θ ← θ + α_a · γ^t · δ · ∇ log π(a|s) ──
            #
            #   For softmax: ∇ log π(a|s) = e_a − π(·|s)
            #   δ replaces G_t from REINFORCE — same gradient, better signal
            #
            p = actor.probs(s)
            onehot = np.zeros(4)
            onehot[a_idx] = 1.0
            score = onehot - p

            actor.theta[s] += lr_actor * (GAMMA ** t) * td_error * score

            s = ns
            t += 1
            if done:
                break

        reward_history.append(total_reward)

        if (ep + 1) % snapshot_interval == 0:
            history.append((
                ep + 1,
                {s: actor.theta[s].copy() for s in actor.theta},
                {s: V[s] for s in V},
            ))

    # ── Derive final greedy policy ──
    policy = {}
    V_array = np.zeros((ROWS, COLS))
    for s in get_valid_states():
        if s == GOAL:
            policy[s] = "★"
        else:
            probs = actor.probs(s)
            policy[s] = ACTIONS[int(np.argmax(probs))]
        V_array[s] = V[s]

    return V_array, policy, history, reward_history
