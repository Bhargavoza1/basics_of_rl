"""
Grid World Environment
======================
5×5 grid with walls, a start cell, and a goal cell.

    S . . . .
    . # . # .
    . . . . .
    . # . # .
    . . . . G

Actions: ↑ ↓ ← →
Rewards: -1 per step, +10 at goal
Hitting a wall or boundary → agent stays in place.
"""

import numpy as np

ROWS = 5
COLS = 5
START = (0, 0)
GOAL = (4, 4)
WALLS = {(1, 1), (1, 3), (3, 1), (3, 3)}

ACTIONS = ["↑", "↓", "←", "→"]
ACTION_DELTAS = {
    "↑": (-1, 0),
    "↓": (1, 0),
    "←": (0, -1),
    "→": (0, 1),
}

GAMMA = 0.99
STEP_REWARD = -1.0
GOAL_REWARD = 10.0


def get_valid_states():
    """Return list of all non-wall cells."""
    return [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) not in WALLS
    ]


def step(state, action):
    """
    Take an action in the grid world.

    Parameters
    ----------
    state : tuple (row, col)
    action : str, one of ACTIONS

    Returns
    -------
    next_state : tuple
    reward     : float
    done       : bool
    """
    dr, dc = ACTION_DELTAS[action]
    nr, nc = state[0] + dr, state[1] + dc

    # Boundary / wall check — stay in place
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or (nr, nc) in WALLS:
        nr, nc = state

    next_state = (nr, nc)

    if next_state == GOAL:
        return next_state, GOAL_REWARD, True

    return next_state, STEP_REWARD, False
