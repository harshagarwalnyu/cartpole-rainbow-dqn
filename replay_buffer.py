"""
replay_buffer.py  —  Prioritized Experience Replay (PER) + N-step returns
==========================================================================
Algorithm
---------
Schaul et al. (2016) "Prioritized Experience Replay" (arXiv:1511.05952)

Components
----------
1. SumTree        – O(log n) priority-based sampling
2. NStepBuffer    – accumulates n-step returns before pushing to main buffer
3. PrioritizedReplayBuffer – main API used by the agent

How PER works
-------------
Each transition is stored with a priority p_i = |TD_error| + eps.
The sampling probability is  P(i) = p_i^alpha / sum(p_j^alpha).
To correct for the non-uniform sampling bias, we weight each gradient
update by  w_i = (1/N * 1/P(i))^beta,  normalised by max(w_i).
beta is annealed from beta_start → 1.0 over training (full correction at
convergence when the policy is near-optimal).
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary tree where leaves hold priorities and parents hold sums.

    capacity: number of leaf nodes (rounded up to next power of 2).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write = 0          # circular write pointer (leaf index)
        self._n_entries = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        """Propagate a priority change up the tree."""
        idx = leaf_idx
        while idx > 1:
            idx >>= 1
            self._tree[idx] += delta

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf index for query value s (tree traversal)."""
        while True:
            left = 2 * idx
            right = left + 1
            if left >= len(self._tree):
                return idx
            if s <= self._tree[left]:
                idx = left
            else:
                s -= self._tree[left]
                idx = right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def add(self, priority: float) -> int:
        """
        Insert a new leaf with given priority.
        Returns the leaf index (for use as data-array index).
        """
        leaf_idx = self._write + self.capacity
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)
        return self._write - 1  # data-array index

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf (leaf_idx is tree index)."""
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, s: float) -> tuple[int, float]:
        """
        Sample one transition by value s ∈ [0, total].
        Returns (data_array_index, priority).
        """
        leaf_idx = self._retrieve(1, s)
        data_idx = leaf_idx - self.capacity
        return data_idx, float(self._tree[leaf_idx])

    def __len__(self) -> int:
        return self._n_entries


# ---------------------------------------------------------------------------
# N-step buffer (accumulates short trajectories)
# ---------------------------------------------------------------------------

class NStepBuffer:
    """
    Collects n consecutive (s, a, r, s', done) tuples and emits a single
    n-step transition:

        G_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1}
        s_n = s_n  (the state n steps ahead)

    When the episode ends before n steps, the buffer is flushed.
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque()

    def push(
        self, s, a, r, s_next, done
    ) -> list[tuple]:
        """
        Add one transition. Returns a list of ready n-step transitions
        (usually 0 or 1 items; all remaining items on done=True).
        """
        self._buf.append((s, a, r, s_next, done))
        ready = []

        if len(self._buf) == self.n:
            ready.append(self._make())
            self._buf.popleft()

        if done:
            # Flush remaining partial transitions
            while self._buf:
                ready.append(self._make())
                self._buf.popleft()

        return ready

    def _make(self) -> tuple:
        """Build one n-step transition from the current buffer."""
        s0, a0, _, _, _ = self._buf[0]
        G = 0.0
        last_s_next = None
        last_done = False
        for i, (_, _, r, s_next, done) in enumerate(self._buf):
            G += (self.gamma ** i) * r
            last_s_next = s_next
            last_done = done
            if done:
                break
        return s0, a0, G, last_s_next, last_done


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity     : int   – max transitions stored
    alpha        : float – priority exponent (0=uniform, 1=full priority)
    beta_start   : float – initial IS-weight exponent (annealed to 1.0)
    beta_frames  : int   – number of frames over which beta reaches 1.0
    eps_priority : float – small constant to ensure non-zero priority
    n_step       : int   – n-step return accumulation
    gamma        : float – discount factor
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps_priority: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps_priority = eps_priority

        self._tree = SumTree(capacity)
        self._data: list = [None] * capacity
        self._max_priority = 1.0
        self._frame = 0

        self._nstep = NStepBuffer(n_step, gamma)

    # ------------------------------------------------------------------
    # Beta annealing
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        frac = min(1.0, self._frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    # ------------------------------------------------------------------
    # Add / Sample
    # ------------------------------------------------------------------

    def push(self, s, a, r, s_next, terminated: bool) -> None:
        """
        Add a raw transition. The n-step buffer will emit ready transitions
        into the SumTree automatically.
        """
        self._frame += 1
        ready = self._nstep.push(s, a, r, s_next, terminated)
        for transition in ready:
            s0, a0, G, sn, dn = transition
            priority = self._max_priority ** self.alpha
            idx = self._tree.add(priority)
            self._data[idx] = (
                np.asarray(s0, dtype=np.float32),
                int(a0),
                float(G),
                np.asarray(sn, dtype=np.float32),
                float(dn),
            )

    def sample(self, batch_size: int):
        """
        Draw a prioritized mini-batch.

        Returns
        -------
        states      : (batch, 4)  float32
        actions     : (batch,)    int64
        returns     : (batch,)    float32  – n-step discounted returns
        next_states : (batch,)    float32
        dones       : (batch,)    float32
        weights     : (batch,)    float32  – IS correction weights
        idxs        : list[int]            – leaf indices for priority update
        """
        n = len(self._tree)
        total = self._tree.total
        segment = total / batch_size
        beta = self.beta

        idxs, priorities, samples = [], [], []
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s_val = np.random.uniform(lo, hi)
            data_idx, priority = self._tree.sample(s_val)
            idxs.append(data_idx + self.capacity)  # tree leaf index
            priorities.append(priority)
            samples.append(self._data[data_idx])

        # IS weights
        min_prob = np.min(priorities) / total
        max_weight = (min_prob * n) ** (-beta)
        probs = np.array(priorities) / total
        weights = ((probs * n) ** (-beta)) / max_weight
        weights = np.asarray(weights, dtype=np.float32)

        s, a, r, s_next, done = zip(*samples)
        return (
            np.array(s,      dtype=np.float32),
            np.array(a,      dtype=np.int64),
            np.array(r,      dtype=np.float32),
            np.array(s_next, dtype=np.float32),
            np.array(done,   dtype=np.float32),
            weights,
            idxs,
        )

    def update_priorities(self, idxs: list[int], td_errors: np.ndarray) -> None:
        """Update tree priorities given new TD errors."""
        for idx, err in zip(idxs, td_errors):
            p = (abs(float(err)) + self.eps_priority) ** self.alpha
            self._max_priority = max(self._max_priority, p)
            self._tree.update(idx, p)

    def __len__(self) -> int:
        return len(self._tree)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size