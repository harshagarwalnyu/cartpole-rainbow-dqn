"""
dqn_agent.py  —  Rainbow-lite agent for CartPole
=================================================
Techniques implemented
----------------------
1. Double DQN         – online net selects action, target net evaluates it.
                        Eliminates Q-overestimation bias.
2. Dueling network    – separate V(s) and A(s,a) streams (in network.py).
3. Prioritized Replay – PER SumTree with n-step returns (replay_buffer.py).
4. Soft target update – Polyak averaging tau=0.005 every gradient step.
                        Smooth target drift instead of hard-copy jumps.
5. Adam + Huber loss  – stable optimisation even with large TD errors.
6. Gradient clipping  – global norm clip=10 prevents exploding gradients.

Expected convergence: ~200–400 episodes to avg100 >= 475.
"""

from __future__ import annotations

import numpy as np

from network import DuelingQNetwork
from replay_buffer import PrioritizedReplayBuffer


class RainbowLiteAgent:
    """
    Parameters
    ----------
    state_dim      : observation dimensionality (4 for CartPole)
    action_dim     : number of discrete actions (2 for CartPole)
    lr             : Adam learning rate
    gamma          : discount factor
    epsilon_start  : initial exploration rate
    epsilon_min    : minimum exploration rate
    epsilon_decay  : multiplicative decay per episode
    batch_size     : PER mini-batch size
    buffer_capacity: max transitions in replay
    tau            : Polyak soft-update coefficient
    n_step         : n-step return length
    alpha          : PER priority exponent
    beta_start     : PER IS-weight initial exponent
    beta_frames    : frames over which beta anneals to 1.0
    hidden         : neurons per hidden layer
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.001,
        epsilon_decay: float = 0.990,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        tau: float = 0.005,
        n_step: int = 3,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        hidden: int = 128,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.n_step = n_step

        # Networks
        self.q_online = DuelingQNetwork(state_dim, action_dim, hidden=hidden, lr=lr)
        self.q_target = DuelingQNetwork(state_dim, action_dim, hidden=hidden, lr=lr)
        self.q_target.copy_from(self.q_online)

        # PER buffer (includes n-step accumulator internally)
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            n_step=n_step,
            gamma=gamma,
        )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        Exploration: random action with probability epsilon.
        Exploitation: argmax Q(s, .) from the online network.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.q_online.forward(state[np.newaxis])  # (1, 2)
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """
        Push one raw transition into the n-step / PER buffer.
        Pass `terminated` (NOT `terminated or truncated`) so truncations
        don't incorrectly zero-out the bootstrap value.
        """
        self.buffer.push(state, action, reward, next_state, terminated)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """
        One Double-DQN gradient step with PER-weighted Huber loss.

        Returns scalar loss (for logging), or None if buffer not ready.
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        s, a, r, s_next, done, weights, tree_idxs = self.buffer.sample(self.batch_size)

        # --- Double DQN target ---
        # 1. Online net picks the BEST ACTION in s_next
        q_online_next = self.q_online.forward(s_next)  # (batch, 2)
        best_actions = np.argmax(q_online_next, axis=1)  # (batch,)

        # 2. Target net EVALUATES that action (decoupled → less overestimation)
        q_target_next = self.q_target.forward(s_next)  # (batch, 2)
        q_next_eval = q_target_next[np.arange(self.batch_size), best_actions]

        # Bellman target (n-step discount already baked into r from NStepBuffer)
        targets = r + (self.gamma**self.n_step) * q_next_eval * (1.0 - done)

        # --- Gradient step + get TD errors for PER update ---
        loss, td_errs = self.q_online.backward(s, targets, a, weights)

        # --- Update priorities in the SumTree ---
        self.buffer.update_priorities(tree_idxs, td_errs)

        # --- Soft target update after every gradient step ---
        self.q_target.soft_update_from(self.q_online, tau=self.tau)

        return loss

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self) -> None:
        """Decay epsilon. Call once per completed episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
