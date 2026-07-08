"""
train.py  —  Rainbow-lite training loop for CartPole-v1
========================================================
Usage
-----
    uv run python train.py             # headless, saves training_curve.png
    uv run python train.py --render    # show pygame window each episode
    uv run python train.py --episodes 600
"""

from __future__ import annotations

import argparse
from collections import deque

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # headless backend — works without a display
import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import RainbowLiteAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPISODES = 600
SOLVE_THRESHOLD = 475  # 100-ep rolling average required to declare solved
WARMUP_STEPS = 1_000  # collect this many transitions before training

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(render: bool = False) -> gym.Env:
    return gym.make("CartPole-v1", render_mode="human" if render else None)


def plot_rewards(rewards: list[float], path: str = "training_curve.png") -> None:
    """Save a clean, annotated learning-curve plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    eps = np.arange(len(rewards))

    ax.plot(eps, rewards, alpha=0.35, color="#74c0fc", linewidth=0.8, label="Episode reward")

    window = 100
    if len(rewards) >= window:
        roll = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(rewards)),
            roll,
            color="#e03131",
            linewidth=2.2,
            label=f"{window}-ep rolling avg",
        )

        # Mark where the agent solved the task
        solved = np.where(roll >= SOLVE_THRESHOLD)[0]
        if len(solved):
            solve_ep = solved[0] + window - 1
            ax.axvline(solve_ep, color="#2f9e44", linestyle="--", alpha=0.8)
            ax.annotate(
                f"Solved ep {solve_ep}",
                xy=(solve_ep, SOLVE_THRESHOLD),
                xytext=(solve_ep + 10, SOLVE_THRESHOLD - 50),
                arrowprops=dict(arrowstyle="->", color="#2f9e44"),
                color="#2f9e44",
                fontsize=9,
            )

    ax.axhline(
        SOLVE_THRESHOLD,
        color="#2f9e44",
        linestyle=":",
        alpha=0.5,
        label=f"Solve threshold ({SOLVE_THRESHOLD})",
    )
    ax.axhline(500, color="#868e96", linestyle=":", alpha=0.4, label="Max reward (500)")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total reward", fontsize=11)
    ax.set_title(
        "Rainbow-lite DQN on CartPole-v1\n"
        "(Double + Dueling + PER + N-step + Soft-target + Adam + Huber)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[PLOT] Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(num_episodes: int = NUM_EPISODES, render: bool = False) -> list[float]:
    env = make_env(render=render)
    agent = RainbowLiteAgent()

    episode_rewards: list[float] = []
    rolling: deque = deque(maxlen=100)
    solved_at: int | None = None
    total_steps = 0

    header = f"{'Ep':>6} | {'Steps':>7} | {'Reward':>7} | {'Avg100':>7} | {'eps':>6} | {'Loss':>9} | {'beta':>5}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        last_loss: float | None = None
        done = False

        # ---- inner step loop ----
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1

            # Store — use terminated only (not truncated) for Bellman correctness
            agent.store(state, action, reward, next_state, terminated)

            # Warm-up: fill buffer before training starts
            if total_steps >= WARMUP_STEPS:
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss

            state = next_state
            total_reward += reward

        # ---- episode bookkeeping ----
        agent.end_episode()
        episode_rewards.append(total_reward)
        rolling.append(total_reward)
        avg100 = float(np.mean(rolling))
        beta = agent.buffer.beta

        loss_str = f"{last_loss:9.4f}" if last_loss is not None else "         -"
        print(
            f"{ep:>6} | {total_steps:>7} | {total_reward:>7.1f} |"
            f" {avg100:>7.1f} | {agent.epsilon:>6.3f} | {loss_str} | {beta:>5.3f}"
        )

        # ---- solved? ----
        if avg100 >= SOLVE_THRESHOLD and solved_at is None:
            solved_at = ep
            print(sep)
            print(
                f"[SOLVED] Episode {ep}  |  avg100 = {avg100:.1f}  |  total steps = {total_steps:,}"
            )
            print(sep)
            break

    env.close()

    if solved_at is None:
        print(sep)
        print(
            f"[WARN] Did not solve in {num_episodes} episodes (best avg100 = {max(np.convolve(episode_rewards, np.ones(min(100, len(episode_rewards))) / min(100, len(episode_rewards)), mode='valid')):.1f})"
        )

    return episode_rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow-lite DQN on CartPole-v1")
    parser.add_argument("--render", action="store_true", help="Render while training")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()

    rewards = train(num_episodes=args.episodes, render=args.render)
    plot_rewards(rewards)
