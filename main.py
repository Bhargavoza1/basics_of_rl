"""
RL Grid World — Main Entry Point
=================================
Run this file to launch the GUI:

    python main.py

Project structure:
    environment.py   — Grid world (states, actions, transitions, rewards)
    mdp.py           — MDP Value Iteration (model-based)
    monte_carlo.py   — First-Visit Monte Carlo Control (model-free)
    reinforce_algo.py — REINFORCE policy gradient (policy-based, no value fn)
    gui.py           — Tkinter visualisation and controls
    main.py          — This file (entry point)

Requirements:
    Python 3.8+
    numpy
    tkinter (included with most Python installations)
"""

import tkinter as tk
from gui import RLGridWorldApp

BG = "#1a1b26"


def main():
    root = tk.Tk()
    root.configure(bg=BG)
    RLGridWorldApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
