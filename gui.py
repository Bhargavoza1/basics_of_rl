"""
GUI — Tkinter Visualisation
============================
Dark-themed grid world viewer with animated training for each algorithm.
Runs algorithms in background threads to keep the UI responsive.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading

from environment import (
    ROWS, COLS, START, GOAL, WALLS, ACTIONS, GAMMA,
    get_valid_states, step,
)
from mdp import value_iteration
from monte_carlo import monte_carlo
from reinforce_algo import reinforce
from td import q_learning
from actor_critic import actor_critic

# ──────────────────────── Layout constants ────────────────────────────────────

CELL = 90
PAD = 30
CANVAS_W = COLS * CELL + 2 * PAD
CANVAS_H = ROWS * CELL + 2 * PAD

# ──────────────────────── Colour palette ──────────────────────────────────────

BG       = "#1a1b26"
GRID_BG  = "#24283b"
WALL_CLR = "#414868"
GOAL_CLR = "#9ece6a"
START_CLR = "#7aa2f7"
TEXT_CLR = "#a9b1d6"
ACCENT   = "#bb9af7"
POS_CLR  = "#73daca"
CARD_BG  = "#1f2335"


def _val_color(v, vmin, vmax):
    """Interpolate from red (#f7768e) to green (#73daca) by normalised value."""
    t = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(0xF7 + t * (0x73 - 0xF7))
    g = int(0x76 + t * (0xDA - 0x76))
    b = int(0x8E + t * (0xCA - 0x8E))
    return f"#{r:02x}{g:02x}{b:02x}"


# ──────────────────────── Main application ────────────────────────────────────

class RLGridWorldApp:
    """Tkinter application with buttons for each algorithm."""

    def __init__(self, root):
        self.root = root
        self.root.title("RL Grid World  —  MDP · MC · TD · REINFORCE · Actor-Critic")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.running = False

        self._setup_styles()
        self._build_widgets()
        self.draw_empty_grid()
        self._draw_info_placeholder()

    # ── Widget construction ───────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",       background=BG)
        s.configure("TLabel",       background=BG, foreground=TEXT_CLR,
                    font=("Helvetica", 10))
        s.configure("Header.TLabel", font=("Helvetica", 14, "bold"),
                    foreground=ACCENT)
        s.configure("TButton",      font=("Helvetica", 10, "bold"),
                    background=ACCENT, foreground="#1a1b26",
                    borderwidth=0, padding=(14, 8))
        s.map("TButton",
              background=[("active", "#9d7cd8"), ("disabled", WALL_CLR)])
        s.configure("Status.TLabel", font=("Helvetica", 9),
                    foreground="#565f89")

    def _build_widgets(self):
        # Header
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=18, pady=(14, 4))
        ttk.Label(top, text="Reinforcement Learning Algorithms",
                  style="Header.TLabel").pack(side="left")

        # Buttons — row 1
        bf = ttk.Frame(self.root)
        bf.pack(fill="x", padx=18, pady=(6, 2))

        self.btn_mdp = ttk.Button(
            bf, text="▶  MDP Value Iteration",
            command=lambda: self._launch("mdp"))
        self.btn_mdp.pack(side="left", padx=(0, 8))

        self.btn_mc = ttk.Button(
            bf, text="▶  Monte Carlo",
            command=lambda: self._launch("mc"))
        self.btn_mc.pack(side="left", padx=(0, 8))

        self.btn_td = ttk.Button(
            bf, text="▶  TD Q-Learning",
            command=lambda: self._launch("td"))
        self.btn_td.pack(side="left", padx=(0, 8))

        self.btn_reset = ttk.Button(
            bf, text="↺  Reset", command=self._reset)
        self.btn_reset.pack(side="right")

        # Buttons — row 2
        bf2 = ttk.Frame(self.root)
        bf2.pack(fill="x", padx=18, pady=(2, 6))

        self.btn_reinforce = ttk.Button(
            bf2, text="▶  REINFORCE",
            command=lambda: self._launch("reinforce"))
        self.btn_reinforce.pack(side="left", padx=(0, 8))

        self.btn_ac = ttk.Button(
            bf2, text="▶  Actor-Critic",
            command=lambda: self._launch("ac"))
        self.btn_ac.pack(side="left", padx=(0, 8))

        # Canvas area
        main = ttk.Frame(self.root)
        main.pack(padx=18, pady=6)

        self.canvas = tk.Canvas(
            main, width=CANVAS_W, height=CANVAS_H,
            bg=GRID_BG, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", padx=(0, 10))

        self.info_canvas = tk.Canvas(
            main, width=340, height=CANVAS_H,
            bg=CARD_BG, highlightthickness=0, bd=0)
        self.info_canvas.pack(side="left")

        # Status bar
        self.status_var = tk.StringVar(
            value="Ready — choose an algorithm above.")
        ttk.Label(self.root, textvariable=self.status_var,
                  style="Status.TLabel").pack(fill="x", padx=18, pady=(2, 12))

    # ── Grid drawing ──────────────────────────────────────────────────────

    def draw_empty_grid(self):
        c = self.canvas
        c.delete("all")
        for r in range(ROWS):
            for col in range(COLS):
                x1, y1 = PAD + col * CELL, PAD + r * CELL
                x2, y2 = x1 + CELL, y1 + CELL

                if (r, col) in WALLS:
                    fill = WALL_CLR
                elif (r, col) == GOAL:
                    fill = "#1e352a"
                elif (r, col) == START:
                    fill = "#1e2a42"
                else:
                    fill = GRID_BG

                c.create_rectangle(x1, y1, x2, y2,
                                   fill=fill, outline="#2f3549", width=1)

                if (r, col) == START:
                    c.create_text(x1 + CELL // 2, y1 + CELL // 2,
                                  text="S", fill=START_CLR,
                                  font=("Helvetica", 16, "bold"))
                elif (r, col) == GOAL:
                    c.create_text(x1 + CELL // 2, y1 + CELL // 2,
                                  text="G", fill=GOAL_CLR,
                                  font=("Helvetica", 16, "bold"))
                elif (r, col) in WALLS:
                    c.create_text(x1 + CELL // 2, y1 + CELL // 2,
                                  text="▨", fill="#565f89",
                                  font=("Helvetica", 18))

    def draw_values_and_policy(self, V, policy):
        """Draw the grid coloured by V with policy arrows."""
        c = self.canvas
        c.delete("all")

        vals = [V[s] for s in get_valid_states() if s != GOAL]
        vmin = min(vals) if vals else 0
        vmax = max(vals) if vals else 1

        for r in range(ROWS):
            for col in range(COLS):
                x1, y1 = PAD + col * CELL, PAD + r * CELL
                x2, y2 = x1 + CELL, y1 + CELL

                if (r, col) in WALLS:
                    c.create_rectangle(x1, y1, x2, y2,
                                       fill=WALL_CLR, outline="#2f3549")
                    c.create_text(x1 + CELL // 2, y1 + CELL // 2,
                                  text="▨", fill="#565f89",
                                  font=("Helvetica", 18))
                    continue

                clr = (GOAL_CLR if (r, col) == GOAL
                       else _val_color(V[r, col], vmin, vmax))
                c.create_rectangle(x1, y1, x2, y2,
                                   fill=clr, outline="#2f3549")

                # Value label
                c.create_text(x1 + CELL // 2, y1 + 18,
                              text=f"{V[r, col]:.1f}", fill="#1a1b26",
                              font=("Helvetica", 9, "bold"))
                # Policy arrow
                arrow = policy.get((r, col), "")
                c.create_text(x1 + CELL // 2, y1 + CELL // 2 + 10,
                              text=arrow, fill="#1a1b26",
                              font=("Helvetica", 22, "bold"))

    # ── Info panel ────────────────────────────────────────────────────────

    def _draw_info_placeholder(self):
        ic = self.info_canvas
        ic.delete("all")
        ic.create_text(170, 30, text="Algorithm Info",
                       fill=ACCENT, font=("Helvetica", 12, "bold"))
        lines = [
            "Grid: 5x5  |  gamma = 0.99",
            "Step reward: -1",
            "Goal reward: +10",
            "",
            "Walls block movement.",
            "Agent stays in place if",
            "it hits a wall or boundary.",
            "",
            "Click an algorithm button",
            "to start training.",
        ]
        for i, line in enumerate(lines):
            ic.create_text(170, 65 + i * 22, text=line,
                           fill=TEXT_CLR, font=("Helvetica", 9))

    def _draw_info(self, name, details, reward_history=None):
        ic = self.info_canvas
        ic.delete("all")
        ic.create_text(170, 25, text=name,
                       fill=ACCENT, font=("Helvetica", 12, "bold"))
        for i, line in enumerate(details):
            ic.create_text(170, 55 + i * 20, text=line,
                           fill=TEXT_CLR, font=("Helvetica", 9))

        if not reward_history or len(reward_history) < 2:
            return

        # ── Smoothed reward chart ──
        window = max(1, len(reward_history) // 50)
        smoothed = []
        for i in range(len(reward_history)):
            lo = max(0, i - window)
            smoothed.append(np.mean(reward_history[lo:i + 1]))

        cx, cy, cw, ch = 30, 200, 280, 180
        ic.create_rectangle(cx, cy, cx + cw, cy + ch,
                            fill="#16161e", outline="#2f3549")
        ic.create_text(cx + cw // 2, cy - 10,
                       text="Episode Reward (smoothed)",
                       fill="#565f89", font=("Helvetica", 8))

        rmin, rmax = min(smoothed), max(smoothed)
        if rmax == rmin:
            rmax = rmin + 1

        n = len(smoothed)
        pts = []
        for i, v in enumerate(smoothed):
            px = cx + 5 + (cw - 10) * i / (n - 1)
            py = cy + ch - 5 - (ch - 10) * (v - rmin) / (rmax - rmin)
            pts.append((px, py))

        for i in range(len(pts) - 1):
            ic.create_line(pts[i][0], pts[i][1],
                           pts[i + 1][0], pts[i + 1][1],
                           fill=POS_CLR, width=1.5)

        ic.create_text(cx + 5, cy + 5, text=f"{rmax:.0f}",
                       anchor="nw", fill="#565f89", font=("Helvetica", 7))
        ic.create_text(cx + 5, cy + ch - 12, text=f"{rmin:.0f}",
                       anchor="nw", fill="#565f89", font=("Helvetica", 7))

    # ── Algorithm orchestration ───────────────────────────────────────────

    def _set_buttons(self, state):
        for b in (self.btn_mdp, self.btn_mc, self.btn_reinforce,
                  self.btn_td, self.btn_ac):
            b.config(state=state)

    def _reset(self):
        if self.running:
            return
        self.draw_empty_grid()
        self._draw_info_placeholder()
        self.status_var.set("Ready — choose an algorithm above.")

    def _launch(self, which):
        if self.running:
            return
        self.running = True
        self._set_buttons("disabled")
        threading.Thread(target=self._run, args=(which,), daemon=True).start()

    def _run(self, which):
        try:
            {"mdp": self._run_mdp,
             "mc": self._run_mc,
             "reinforce": self._run_reinforce,
             "td": self._run_td,
             "ac": self._run_ac}[which]()
        finally:
            self.running = False
            self.root.after(0, lambda: self._set_buttons("normal"))

    # ── MDP ───────────────────────────────────────────────────────────────

    def _run_mdp(self):
        self.root.after(0, lambda: self.status_var.set(
            "Running MDP Value Iteration ..."))
        V, policy, history = value_iteration()

        # Animate each sweep
        for i, v_snap in enumerate(history):
            p_snap = self._greedy_policy(v_snap)
            self.root.after(0, self.draw_values_and_policy, v_snap, p_snap)
            time.sleep(max(0.02, 0.15 - i * 0.005))

        self.root.after(0, self.draw_values_and_policy, V, policy)
        self.root.after(0, self._draw_info, "MDP Value Iteration", [
            "Method: Bellman optimality backup",
            f"Converged in {len(history)} iterations",
            f"theta = 0.0001  |  gamma = {GAMMA}",
            "",
            "V(s) = max_a  Sum P(s'|s,a)",
            "        x [R + gamma V(s')]",
            "",
            "Deterministic transitions.",
            "Policy = greedy over V.",
        ])
        self.root.after(0, lambda: self.status_var.set(
            f"MDP converged in {len(history)} iterations."))

    # ── Monte Carlo ───────────────────────────────────────────────────────

    def _run_mc(self):
        n_ep = 3000
        self.root.after(0, lambda: self.status_var.set(
            f"Running Monte Carlo ({n_ep} episodes) ..."))
        V, policy, history = monte_carlo(n_episodes=n_ep)

        for ep_num, Q_snap in history:
            V_snap = np.zeros((ROWS, COLS))
            p_snap = {}
            for s in get_valid_states():
                if s == GOAL:
                    p_snap[s] = "★"
                else:
                    p_snap[s] = ACTIONS[int(np.argmax(Q_snap[s]))]
                    V_snap[s] = np.max(Q_snap[s])
            self.root.after(0, self.draw_values_and_policy, V_snap, p_snap)
            self.root.after(0, lambda e=ep_num: self.status_var.set(
                f"Monte Carlo — episode {e}/{n_ep}"))
            time.sleep(0.08)

        self.root.after(0, self.draw_values_and_policy, V, policy)
        self.root.after(0, self._draw_info, "Monte Carlo Control", [
            "Method: First-Visit MC Control",
            f"Episodes: {n_ep}  |  eps = 0.2",
            f"gamma = {GAMMA}",
            "",
            "Q(s,a) <- average(returns)",
            "eps-greedy improvement",
            "",
            "Model-free: learns from",
            "sampled complete episodes.",
        ])
        self.root.after(0, lambda: self.status_var.set(
            f"Monte Carlo finished ({n_ep} episodes)."))

    # ── REINFORCE ─────────────────────────────────────────────────────────

    def _run_reinforce(self):
        n_ep = 4000
        self.root.after(0, lambda: self.status_var.set(
            f"Running REINFORCE ({n_ep} episodes) ..."))
        V, policy, history, rewards = reinforce(n_episodes=n_ep, lr=0.01)

        for ep_num, theta_snap in history:
            V_snap = np.zeros((ROWS, COLS))
            p_snap = {}
            for s in get_valid_states():
                if s == GOAL:
                    p_snap[s] = "★"
                    continue
                logits = theta_snap[s] - np.max(theta_snap[s])
                probs = np.exp(logits) / np.exp(logits).sum()
                p_snap[s] = ACTIONS[int(np.argmax(probs))]
                V_snap[s] = np.max(probs)
            self.root.after(0, self.draw_values_and_policy, V_snap, p_snap)
            self.root.after(0, lambda e=ep_num: self.status_var.set(
                f"REINFORCE — episode {e}/{n_ep}"))
            time.sleep(0.08)

        self.root.after(0, self.draw_values_and_policy, V, policy)
        self.root.after(0, self._draw_info,
                        "REINFORCE (Policy Gradient)", [
                            "Method: REINFORCE (Williams 1992)",
                            f"Episodes: {n_ep}  |  lr = 0.01",
                            f"gamma = {GAMMA}  |  No baseline",
                            "",
                            "theta <- theta + a * gamma^t",
                            "         * G_t * grad log pi",
                            "Softmax tabular policy.",
                            "",
                            "Pure policy gradient,",
                            "no value function used.",
                        ], rewards)
        self.root.after(0, lambda: self.status_var.set(
            f"REINFORCE finished ({n_ep} episodes)."))

    # ── TD Q-Learning ─────────────────────────────────────────────────────

    def _run_td(self):
        n_ep = 2000
        self.root.after(0, lambda: self.status_var.set(
            f"Running TD Q-Learning ({n_ep} episodes) ..."))
        V, policy, history, rewards = q_learning(n_episodes=n_ep)

        for ep_num, Q_snap in history:
            V_snap = np.zeros((ROWS, COLS))
            p_snap = {}
            for s in get_valid_states():
                if s == GOAL:
                    p_snap[s] = "★"
                else:
                    p_snap[s] = ACTIONS[int(np.argmax(Q_snap[s]))]
                    V_snap[s] = np.max(Q_snap[s])
            self.root.after(0, self.draw_values_and_policy, V_snap, p_snap)
            self.root.after(0, lambda e=ep_num: self.status_var.set(
                f"TD Q-Learning — episode {e}/{n_ep}"))
            time.sleep(0.08)

        self.root.after(0, self.draw_values_and_policy, V, policy)
        self.root.after(0, self._draw_info,
                        "TD Q-Learning", [
                            "Method: Off-policy TD Control",
                            f"Episodes: {n_ep}  |  alpha = 0.1",
                            f"gamma = {GAMMA}  |  eps = 0.2",
                            "",
                            "Q(s,a) <- Q(s,a) + alpha *",
                            "  [r + gamma max Q(s',a')",
                            "   - Q(s,a)]",
                            "",
                            "Bootstraps: updates every",
                            "step, not end of episode.",
                        ], rewards)
        self.root.after(0, lambda: self.status_var.set(
            f"TD Q-Learning finished ({n_ep} episodes)."))

    # ── Actor-Critic ──────────────────────────────────────────────────────

    def _run_ac(self):
        n_ep = 2000
        self.root.after(0, lambda: self.status_var.set(
            f"Running Actor-Critic ({n_ep} episodes) ..."))
        V, policy, history, rewards = actor_critic(n_episodes=n_ep)

        for ep_num, theta_snap, v_snap in history:
            V_snap = np.zeros((ROWS, COLS))
            p_snap = {}
            for s in get_valid_states():
                if s == GOAL:
                    p_snap[s] = "★"
                    continue
                logits = theta_snap[s] - np.max(theta_snap[s])
                probs = np.exp(logits) / np.exp(logits).sum()
                p_snap[s] = ACTIONS[int(np.argmax(probs))]
                V_snap[s] = v_snap[s]
            self.root.after(0, self.draw_values_and_policy, V_snap, p_snap)
            self.root.after(0, lambda e=ep_num: self.status_var.set(
                f"Actor-Critic — episode {e}/{n_ep}"))
            time.sleep(0.08)

        self.root.after(0, self.draw_values_and_policy, V, policy)
        self.root.after(0, self._draw_info,
                        "Actor-Critic", [
                            "Method: One-Step Actor-Critic",
                            f"Episodes: {n_ep}",
                            f"lr_actor=0.01  lr_critic=0.1",
                            f"gamma = {GAMMA}",
                            "",
                            "delta = r + gamma V(s') - V(s)",
                            "Critic: V(s) += a_c * delta",
                            "Actor:  theta += a_a * delta",
                            "         * grad log pi(a|s)",
                            "",
                            "TD + Policy Gradient combined.",
                        ], rewards)
        self.root.after(0, lambda: self.status_var.set(
            f"Actor-Critic finished ({n_ep} episodes)."))

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _greedy_policy(V):
        """Extract greedy policy from a value array."""
        policy = {}
        for s in get_valid_states():
            if s == GOAL:
                policy[s] = "★"
                continue
            best_a, best_v = ACTIONS[0], -1e9
            for a in ACTIONS:
                ns, r, _ = step(s, a)
                v = r + GAMMA * V[ns]
                if v > best_v:
                    best_v, best_a = v, a
            policy[s] = best_a
        return policy