# Reinforcement Learning Grid World

A learning project built with [Claude](https://claude.ai) to understand the
five foundational RL algorithms by implementing them from scratch on a simple
grid world — with animated training, live policy updates, and reward charts.

The code was generated with AI assistance. The understanding is mine. See
[Learning Notes](#learning-notes) for the key insights I picked up along the
way.

```
python main.py
```

Requires **Python 3.8+**, **numpy**, and **tkinter** (ships with most Python
installations).

---

## The Environment

The agent lives on a **5 × 5 grid** and must find the shortest path from the
top-left corner to the bottom-right corner while avoiding walls.

```
 S   .   .   .   .          S = Start  (0,0)
 .   #   .   #   .          G = Goal   (4,4)
 .   .   .   .   .          # = Wall   (impassable)
 .   #   .   #   .          . = Empty cell
 .   .   .   .   G
```

**Actions.** At every step the agent picks one of four directions: ↑ ↓ ← →.
If the move would land on a wall or outside the grid, the agent stays where it
is.

**Rewards.** Every step costs −1, encouraging the agent to reach the goal
quickly. Arriving at the goal gives +10 and ends the episode.

**Discount factor (γ = 0.99).** Future rewards are worth slightly less than
immediate ones, which makes the agent prefer shorter paths.

These definitions live in `environment.py`. Every algorithm imports the same
`step(state, action)` function, so they all operate on identical dynamics.

---

## Algorithm 1 — MDP Value Iteration

**File:** `mdp.py`

### What it is

Value Iteration is a **model-based** dynamic programming method. It assumes
the agent has complete knowledge of the environment — it knows exactly what
happens when it takes each action in each state (the transition function and
rewards). This is called having access to the **model** (the MDP).

### How it works

The algorithm maintains a table **V(s)** — one number for every state — that
represents how good it is to be in that state under the best possible
behaviour. It starts at zero everywhere and repeatedly improves the estimates
using the **Bellman optimality equation**:

```
V(s) ← max over all actions a of [ R(s,a) + γ · V(next state) ]
```

In plain terms: the value of a state equals the best single-step reward the
agent can get, plus the (discounted) value of wherever that step leads. Because
the grid world is deterministic (each action has exactly one outcome), there is
no probability sum — the "max" over actions is all that matters.

The algorithm sweeps over every state, updates V, and checks how much anything
changed. When the largest change (delta) drops below a tiny threshold (θ =
0.0001), it stops — the values have converged.

### Extracting the policy

Once V has converged, the **greedy policy** falls out directly:

```
π(s) = the action a that maximises  R(s,a) + γ · V(next state)
```

At every state the agent simply looks one step ahead, picks the action that
leads to the highest-valued neighbour, and that is the optimal policy.

### In the GUI

You will see the grid cells change colour sweep by sweep as V converges (about
9 iterations on this grid). Red means low value, green means high value.
Arrows show the greedy policy at each snapshot.

### Key properties

- **Guaranteed optimal** for finite MDPs.
- Requires the full model (transitions and rewards).
- Very fast on small state spaces; impractical when the state space is large.

---

## Algorithm 2 — Monte Carlo Control

**File:** `monte_carlo.py`

### What it is

Monte Carlo (MC) is a **model-free** algorithm. The agent does not know the
transition function — it learns purely by generating episodes (sequences of
state → action → reward) and observing what happens.

### How it works

The algorithm maintains a table **Q(s, a)** — one number for every
state-action pair — estimating the expected return of taking action `a` in
state `s` and following the current policy afterwards.

Training loop (repeated for 3 000 episodes):

1. **Generate an episode.** Start at S, pick actions according to the current
   ε-greedy policy, and play until the agent reaches G or a step limit.

2. **Compute returns.** Walk backwards through the episode. At each time step
   `t`, accumulate the discounted return:

   ```
   G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + ...
   ```

3. **First-visit update.** For each (state, action) pair, only use the return
   from the *first* time that pair appears in the episode. Average all such
   returns seen so far:

   ```
   Q(s, a) ← mean of all first-visit returns for (s, a)
   ```

4. **Policy improvement.** Update the policy to be ε-greedy with respect to
   the new Q values:

   ```
   π(a|s) = 1 − ε + ε/|A|    if a = argmax Q(s, ·)
            ε / |A|            otherwise
   ```

   Here ε = 0.2, so the best action gets 85% probability and the other three
   share the remaining 15%. This ensures the agent keeps exploring.

### In the GUI

The grid animates every ~100 episodes, showing Q-derived values and the
current greedy policy. Early on the arrows may look erratic; by episode 3 000
they should closely match the MDP solution.

### Key properties

- **Model-free** — no knowledge of transitions needed.
- Needs **complete episodes** (must reach a terminal state to compute returns).
- Unbiased estimates (no bootstrapping), but high variance.
- Slower to converge than value iteration because it relies on sampling.

---

## Algorithm 3 — REINFORCE (Policy Gradient)

**File:** `reinforce_algo.py`

### What it is

REINFORCE (Williams, 1992) is a **policy-based** method. Instead of learning a
value function and deriving a policy from it, REINFORCE directly parameterises
and optimises the policy itself. There is **no value function** anywhere in
this implementation.

### The policy

The policy is a **tabular softmax**. For every state `s`, there is a parameter
vector θ_s with one entry per action. The probability of picking action `a` in
state `s` is:

```
π(a | s) = exp(θ_{s,a}) / Σ_{a'} exp(θ_{s,a'})
```

At initialisation all θ are zero, so all actions are equally likely (uniform
random). Training shifts the parameters to make good actions more probable.

### How it works

Training loop (repeated for 4 000 episodes):

1. **Generate an episode.** Start at S, sample actions from π(·|s), record the
   trajectory of (state, action, reward) tuples.

2. **Compute returns.** Same backward accumulation as Monte Carlo:

   ```
   G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + ...
   ```

3. **Policy gradient update.** For every step `t` in the episode, nudge θ in
   the direction that makes the chosen action more likely, scaled by how good
   the outcome was:

   ```
   θ_{s,·} ← θ_{s,·} + α · γ^t · G_t · ∇_θ log π(a_t | s_t)
   ```

   For a softmax policy the gradient of the log-probability (the **score
   function**) has a clean form:

   ```
   ∇_θ log π(a | s) = e_a − π(· | s)
   ```

   where `e_a` is a one-hot vector for the chosen action. In words: increase
   the logit of the action you took and decrease the logits of every other
   action, proportionally to their current probabilities.

   The factor `γ^t` down-weights updates from later time steps (discount
   weighting), and `G_t` scales the update by the return — actions that led to
   high returns get reinforced more.

### No baseline

The classic REINFORCE paper also describes a **baseline** (typically a learned
value function V(s)) subtracted from G_t to reduce variance. This
implementation deliberately omits the baseline to keep it a pure policy-only
method, as requested. The trade-off is higher variance in the gradient
estimates, which means slower and noisier learning.

### In the GUI

The grid animates periodically. The right-hand panel shows a **smoothed reward
curve** — you can watch total episode reward climb from heavily negative (agent
wanders) to near-optimal (agent takes the shortest path). Because there is no
baseline, the curve will be noisy.

### Key properties

- **Policy-based** — optimises the policy directly, no value function.
- **Model-free** — learns from sampled episodes.
- Can naturally handle stochastic policies and continuous action spaces (though
  this demo uses a discrete grid).
- High variance without a baseline; converges more slowly than MC or value
  iteration on this problem.

---

## Algorithm 4 — TD Q-Learning

**File:** `td.py`

### What it is

Q-Learning (Watkins, 1989) is a **model-free, value-based** algorithm — like
Monte Carlo in that it learns a Q(s,a) table without knowing the environment's
transitions. But it introduces a powerful idea that Monte Carlo lacks:
**bootstrapping**.

### The problem with Monte Carlo

Monte Carlo has to wait until an episode ends to learn anything. If the agent
wanders for 150 steps before reaching the goal, it can't update Q until all
150 steps are done. This is wasteful — surely after a single step the agent has
already gained *some* information about the current state.

### How TD fixes this

TD (Temporal Difference) learns **after every single step**. The moment the
agent takes an action and sees the next state, it updates Q immediately. The
key insight is: instead of waiting for the real return G (which requires the
full episode), TD *estimates* the return using the Q values it already has.

After taking action `a` in state `s`, getting reward `r`, and landing in `s'`:

```
TD target = r + γ · max_a' Q(s', a')
TD error  = TD target − Q(s, a)
Q(s, a)  ← Q(s, a) + α · TD error
```

The **TD target** is the agent's best guess of the return: the immediate
reward `r` plus the discounted value of the best thing it can do from the next
state. This "best thing from next state" part is the **bootstrap** — it uses
its own current estimates to improve itself.

The **TD error** (δ) measures surprise: how different the observed transition
was from what Q predicted. If the target is higher than expected, Q goes up.
If lower, Q goes down. The learning rate α (0.1) controls how big each step
is.

### Why "off-policy"?

Q-Learning uses `max` over the next state's actions in the update, meaning it
always learns about the *best possible* policy, even though the agent might be
exploring with random actions (ε-greedy). The behaviour policy (what the agent
actually does) and the target policy (what it's learning about) are different —
that's what "off-policy" means.

Compare to Monte Carlo, which is on-policy: it evaluates whichever policy it
is currently following.

### TD vs Monte Carlo — the core trade-off

Monte Carlo uses the **real return** G — the actual sum of rewards the agent
collected. This is unbiased (over enough episodes, it converges to the true
value) but high variance (each episode's path is different, so G fluctuates a
lot).

TD uses a **bootstrapped estimate** — `r + γ · max Q(s', a')`. This has lower
variance (it's a single step, not an entire episode's worth of randomness) but
introduces **bias** because the Q estimate used for bootstrapping might be
wrong early in training. As training continues, Q improves, and the bias
shrinks.

In practice, TD's lower variance usually wins: it converges faster than Monte
Carlo on most problems. In our grid world, TD Q-Learning needs about 2,000
episodes vs Monte Carlo's 3,000.

### In the GUI

The grid animates periodically, and the right panel shows a smoothed reward
curve. You'll notice TD converges faster and more smoothly than Monte Carlo —
that's the variance reduction from bootstrapping at work.

### Key properties

- **Model-free** — no knowledge of transitions needed.
- **Bootstraps** — updates after each step using estimated values.
- **Off-policy** — learns optimal policy while following exploratory policy.
- Lower variance than MC, slight bias early on.
- Does not need complete episodes (can learn from incomplete runs).

---

## Algorithm 5 — Actor-Critic

**File:** `actor_critic.py`

### What it is

Actor-Critic is where the two branches of RL merge. It has two components
working together:

The **Critic** is a state-value table V(s), updated with TD learning — the
same idea from Algorithm 4 (TD Q-Learning), except it tracks V(s) instead of
Q(s,a).

The **Actor** is a softmax policy π(a|s) parameterised by θ — the same
structure from Algorithm 3 (REINFORCE).

The critic tells the actor how good its actions are. The actor uses that
feedback to improve. Neither works alone the way they do here — the
combination is more than the sum of its parts.

### The problem with REINFORCE

Recall that REINFORCE updates the policy using the full return G_t:

```
θ ← θ + α · γᵗ · G_t · ∇ log π(a|s)
```

G_t is the sum of all rewards from step t to the end of the episode. This has
two problems: (1) it requires waiting until the episode finishes, and (2) G_t
has high variance because it includes the randomness of every future step.

### How Actor-Critic fixes this

Actor-Critic replaces G_t with the **TD error** δ — a one-step advantage
estimate from the critic:

```
δ = r + γ · V(s') − V(s)
```

This single number answers: "was this action better or worse than expected?"
If the agent lands in a state that the critic values highly (V(s') is large),
δ is positive — the action was good. If V(s') is low, δ is negative — the
action was bad.

The full update rules, applied **after every single step**:

```
δ         = r + γ · V(s') − V(s)              (TD error = advantage)
V(s)     ← V(s)   + α_c · δ                  (critic learns via TD)
θ_{s,·} ← θ_{s,·} + α_a · γᵗ · δ · ∇ log π(a|s)   (actor learns via policy gradient)
```

The critic update is pure TD learning (Chapter 6 of Sutton & Barto). The actor
update is the same policy gradient as REINFORCE, but with δ replacing G_t.

### Why this is better

Compare what each algorithm needs to update the policy:

```
REINFORCE:     needs G_t → must finish entire episode → high variance
Actor-Critic:  needs δ   → updates after one step    → low variance
```

δ is based on a single transition (s, a, r, s'), not a whole trajectory. Less
randomness means lower variance. The price is bias — the critic's V might be
inaccurate early on — but that bias shrinks as V improves.

### The advantage interpretation

The TD error δ = r + γV(s') − V(s) is an estimate of the **advantage**
A(s, a) = Q(s, a) − V(s). Here is why:

- Q(s, a) ≈ r + γ · V(s') — the expected return of taking action a, then
  following the policy (approximated by the critic's estimate of the next
  state).
- V(s) — the expected return of following the policy from s in general.
- The difference tells you how much better (or worse) action a was compared to
  the average action.

This is the same "advantage" concept discussed earlier, and here it falls out
naturally from the TD error.

### In the GUI

The grid animates periodically, showing V(s) values from the critic and arrows
from the actor. The right panel shows a smoothed reward curve. You should see
faster and smoother convergence than REINFORCE — that is the variance reduction
from the critic at work.

### Key properties

- **Model-free** — learns from experience, no transition model needed.
- **Online** — updates after every step, no need for complete episodes.
- **Lower variance** than REINFORCE (TD error replaces full return).
- **Slight bias** early on (critic bootstraps from imperfect estimates).
- Foundation of all modern deep RL (A2C, A3C, PPO, SAC build on this).

---

## Comparing the Five Algorithms

```
                    MDP Value Iter.   Monte Carlo       TD Q-Learning     REINFORCE         Actor-Critic
                    ───────────────   ─────────────     ─────────────     ──────────────    ──────────────
Needs model?        Yes               No                No                No                No
Learns what?        V(s)              Q(s,a)            Q(s,a)            Policy θ          Policy θ + V(s)
Policy derived?     From V (greedy)   From Q (ε-greedy) From Q (ε-greedy) Directly (softmax)Directly (softmax)
Uses bootstrapping? Yes               No                Yes               No                Yes (critic)
Episode required?   No (sweeps)       Yes (complete)    No (step-by-step) Yes (complete)    No (step-by-step)
Convergence         Very fast (~9)    Moderate (~3000)  Fast (~2000)      Slow (~4000)      Fast (~2000)
Variance            None              Medium            Low               High              Low
```

---

## Project Structure

```
rl_gridworld/
├── main.py            Entry point — run this
├── environment.py     Grid world: states, walls, step(), rewards
├── mdp.py             Value Iteration (model-based, dynamic programming)
├── monte_carlo.py     First-Visit MC Control (model-free, episode-based)
├── td.py              Q-Learning (model-free, bootstrapping, step-by-step)
├── reinforce_algo.py  REINFORCE policy gradient (policy-based, no value fn)
├── actor_critic.py    Actor-Critic (policy + value, TD advantage)
└── gui.py             Tkinter GUI: grid canvas, info panel, reward chart
```

Each algorithm file is self-contained. It imports only `environment.py` and
exposes a single function (`value_iteration()`, `monte_carlo()`,
`q_learning()`, `reinforce()`, or `actor_critic()`) that returns the learned
value table, policy, and training history. The GUI imports all five and runs
whichever one you click.

---

## Tuning Parameters

All defaults are set in the algorithm files. You can change them by editing the
function calls in `gui.py`:

| Parameter       | Where              | Default | Effect                              |
|-----------------|--------------------|---------|-------------------------------------|
| `theta`         | `mdp.py`           | 0.0001  | Convergence threshold for VI        |
| `n_episodes`    | `monte_carlo.py`   | 3000    | More episodes → better Q estimates  |
| `epsilon`       | `monte_carlo.py`   | 0.2     | Exploration rate (0 = greedy)       |
| `n_episodes`    | `td.py`            | 2000    | More episodes → better convergence  |
| `alpha`         | `td.py`            | 0.1     | TD learning rate (step size)        |
| `epsilon`       | `td.py`            | 0.2     | Exploration rate for ε-greedy       |
| `n_episodes`    | `reinforce_algo.py`| 4000    | More episodes → smoother learning   |
| `lr`            | `reinforce_algo.py`| 0.01    | Step size for gradient updates      |
| `n_episodes`    | `actor_critic.py`  | 2000    | More episodes → better convergence  |
| `lr_actor`      | `actor_critic.py`  | 0.01    | Actor learning rate (policy)        |
| `lr_critic`     | `actor_critic.py`  | 0.1     | Critic learning rate (value)        |
| `GAMMA`         | `environment.py`   | 0.99    | Discount factor (shared by all)     |
| `STEP_REWARD`   | `environment.py`   | −1.0    | Per-step penalty                    |
| `GOAL_REWARD`   | `environment.py`   | +10.0   | Terminal reward at goal             |

---

## Learning Notes

Key insights I picked up while building and studying these algorithms:

### MDP doesn't play the game

Value Iteration never enters the grid. It sits outside, looks at every cell,
and asks the model "what would happen if I did this?" It computes exact values
through pure math — no episodes, no randomness, no starting position. That's
why it converges in 9 sweeps. The price is it needs the full model (transition
function + rewards) up front.

### There is no backpropagation in MDP

Values don't "backtrack" from goal to start. Each sweep updates every cell by
looking one step ahead at its neighbours. The useful information from the goal
ripples outward one layer per sweep — not because the algorithm is directional,
but because neighbours read each other's values. After enough sweeps the ripple
reaches every cell.

### Monte Carlo and REINFORCE both use full G_t

Both generate complete episodes and compute the full discounted return
G_t = r_0 + γr_1 + γ²r_2 + ... by walking the episode backward. The only
difference is what they do with G_t:

- **Monte Carlo** → averages G_t into a Q(s,a) table (statistics)
- **REINFORCE** → uses G_t to push policy parameters θ via gradient ascent

Same input, different destination.

### Monte Carlo has one real table, not two

The Q table is the brain. The "policy" is just a wrapper that reads Q and
picks the highest-valued action with ε-greedy. Delete the policy object and
call `argmax(Q[s])` directly — same result. In REINFORCE it's the opposite:
θ (the policy) is the only thing that exists.

### TD's advantage over Monte Carlo is when it learns

Monte Carlo: play the full episode, then update everything at the end.
TD: update after every single step, while still playing. Bootstrapping
(`r + γ max Q(s')`) lets TD estimate the return without waiting. Less
randomness per update means lower variance, which means faster convergence.

### Actor-Critic is where everything meets

The critic is TD applied to V(s). The actor is REINFORCE applied to θ. The
TD error δ = r + γV(s') − V(s) is the estimated advantage — it tells the
actor "was that action better or worse than expected?" This replaces the
noisy full return G_t from REINFORCE with a low-variance one-step signal.
The shared δ variable is literally the bridge between the two.

### Every modern RL method is a flavour of these five ideas

PPO, SAC, A3C, DDPG — they all trace back to the concepts here. PPO is
Actor-Critic + GAE (blended n-step advantages) + clipped policy updates.
The foundation never changes; the refinements are about stability and scale.

---

## How the Algorithms Connect

```
                    Model-Based          Model-Free
                    ───────────          ──────────
Value-based:        MDP Value Iter. ──→  Monte Carlo ──→ TD Q-Learning
                                                              │
                                                              │ (TD error δ)
                                                              ▼
Policy-based:                            REINFORCE ────→ Actor-Critic
                                         (full G_t)      (δ replaces G_t)
```

The left-to-right progression removes limitations:
- MDP needs the model → MC removes that
- MC needs full episodes → TD removes that
- REINFORCE has high variance → Actor-Critic reduces it using TD's critic

---

## What's Next

These five algorithms are all **tabular** — they store values in tables, one
entry per state (or state-action pair). This works for a 5×5 grid with 21
states, but breaks down when the state space is large or continuous.

The next step is **deep RL**: replacing tables with neural networks. The
algorithms stay the same conceptually — DQN is Q-Learning with a neural net,
PPO is Actor-Critic with clipped updates — but the implementation changes
significantly.

For that you would need:

- **Gymnasium** (OpenAI Gym) — environments like CartPole (4 continuous state
  variables), LunarLander (8 variables), or Atari (pixel images as states)
- **PyTorch** — to build the neural networks that approximate Q or π
- **Stable Baselines3** (optional) — pre-built deep RL implementations to
  compare against

The learning path continues:

```
Tabular (this project)
    ↓
DQN         ← Q-Learning + neural network + experience replay
    ↓
PPO         ← Actor-Critic + GAE + clipped policy updates
    ↓
SAC         ← Actor-Critic + entropy bonus (continuous actions)
```

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) —
  Chapters 4 (DP), 5 (Monte Carlo), 6 (TD Learning), 13 (Policy Gradient).
- Watkins, C. J. C. H. (1989). Learning from delayed rewards. PhD thesis,
  Cambridge University.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for
  connectionist reinforcement learning. *Machine Learning*, 8, 229–256.