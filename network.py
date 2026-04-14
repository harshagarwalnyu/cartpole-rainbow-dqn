"""
network.py  —  Dueling Q-Network (NumPy)
=========================================
Architecture
------------
Input (4) ──► Shared MLP  ──► Value stream  V(s):  scalar
                           └──► Advantage stream A(s,a): 2-dim

Q(s,a) = V(s) + [ A(s,a) - mean_a(A(s,a)) ]

Why Dueling?
  • The agent can learn *which states are valuable* (V) without having to
    learn the effect of each action (A) in every state.
  • Leads to faster, more stable convergence — especially powerful when
    many actions have similar expected value (e.g. left≈right while pole
    is nearly vertical).

Optimizer: Adam with bias-correction.
Loss:      Huber (smooth-L1), robust to large TD-error outliers.
           δ = 1.0  ⟹  L1 for |err|>1, L2 for |err|≤1.
Gradient:  Global-norm clipping at 10.0.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)

def _huber(errors: np.ndarray, delta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Huber loss and its gradient w.r.t. errors.

    Returns
    -------
    loss_per_sample : (batch,)
    grad_per_sample : (batch,)  — d(Huber)/d(error)
    """
    abs_err = np.abs(errors)
    quadratic = np.minimum(abs_err, delta)
    loss = 0.5 * quadratic ** 2 + delta * (abs_err - quadratic)
    grad = np.where(abs_err <= delta, errors, delta * np.sign(errors))
    return loss, grad

# ---------------------------------------------------------------------------
# Parameter block helper
# ---------------------------------------------------------------------------

class _Params:
    """Holds one weight matrix + bias + Adam moments."""

    def __init__(self, shape_in: int, shape_out: int, gain: float = 2.0) -> None:
        scale = np.sqrt(gain / shape_in)
        self.W = np.random.randn(shape_in, shape_out).astype(np.float32) * scale
        self.b = np.zeros(shape_out, dtype=np.float32)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def update(
        self,
        dW: np.ndarray,
        db: np.ndarray,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        t: int,
    ) -> None:
        """One Adam step (in-place)."""
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * dW ** 2
        self.W -= lr * (self.mW / bc1) / (np.sqrt(self.vW / bc2) + eps)
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * db ** 2
        self.b -= lr * (self.mb / bc1) / (np.sqrt(self.vb / bc2) + eps)

    def clone(self) -> "_Params":
        p = _Params.__new__(_Params)
        p.W = self.W.copy()
        p.b = self.b.copy()
        p.mW = self.mW.copy()
        p.vW = self.vW.copy()
        p.mb = self.mb.copy()
        p.vb = self.vb.copy()
        return p

# ---------------------------------------------------------------------------
# Dueling Q-Network
# ---------------------------------------------------------------------------

class DuelingQNetwork:
    """
    Dueling Double-DQN Q-network.

    Parameters
    ----------
    state_dim  : int   – observation size (4 for CartPole)
    action_dim : int   – number of actions (2 for CartPole)
    hidden     : int   – neurons per shared hidden layer
    lr         : float – Adam learning rate
    huber_delta: float – Huber loss threshold
    clip_norm  : float – global gradient-norm clip value
    """

    BETA1 = 0.9
    BETA2 = 0.999
    EPS_ADAM = 1e-8

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden: int = 128,
        lr: float = 5e-4,
        huber_delta: float = 1.0,
        clip_norm: float = 10.0,
    ) -> None:
        self.action_dim = action_dim
        self.lr = lr
        self.huber_delta = huber_delta
        self.clip_norm = clip_norm
        self._t = 0  # Adam step counter

        # Shared trunk (2 hidden layers)
        self.fc1 = _Params(state_dim, hidden)
        self.fc2 = _Params(hidden, hidden)

        # Value stream:     hidden → 64 → 1
        self.v1 = _Params(hidden, 64)
        self.v2 = _Params(64, 1)

        # Advantage stream: hidden → 64 → action_dim
        self.a1 = _Params(hidden, 64)
        self.a2 = _Params(64, action_dim)

        # Cached forward activations (set during forward pass)
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, s: np.ndarray) -> np.ndarray:
        """
        s : (batch, 4)  →  Q-values : (batch, 2)
        """
        # --- Shared trunk ---
        z1 = s @ self.fc1.W + self.fc1.b
        a1 = _relu(z1)
        z2 = a1 @ self.fc2.W + self.fc2.b
        a2 = _relu(z2)

        # --- Value stream ---
        zv1 = a2 @ self.v1.W + self.v1.b
        av1 = _relu(zv1)
        V = av1 @ self.v2.W + self.v2.b          # (batch, 1)

        # --- Advantage stream ---
        za1 = a2 @ self.a1.W + self.a1.b
        aa1 = _relu(za1)
        A = aa1 @ self.a2.W + self.a2.b          # (batch, action_dim)

        # --- Dueling aggregation ---
        Q = V + (A - A.mean(axis=1, keepdims=True))

        # Cache for backprop
        self._cache = dict(
            s=s, z1=z1, a1=a1, z2=z2, a2=a2,
            zv1=zv1, av1=av1, V=V,
            za1=za1, aa1=aa1, A=A, Q=Q,
        )
        return Q

    # ------------------------------------------------------------------
    # Backward pass — IS-weighted Huber loss
    # ------------------------------------------------------------------

    def backward(
        self,
        s: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,          # IS weights from PER, shape (batch,)
    ) -> tuple[float, np.ndarray]:
        """
        Weighted Huber loss over chosen actions only.

        Returns
        -------
        loss    : float        – scalar mean loss (for logging)
        td_errs : (batch,)     – raw TD errors (for PER priority update)
        """
        batch = s.shape[0]
        Q = self.forward(s)

        td_errs = Q[np.arange(batch), actions] - targets     # (batch,)
        huber_loss, huber_grad = _huber(td_errs, self.huber_delta)
        loss = float(np.mean(weights * huber_loss))

        # Gradient into Q-values:  only the taken action column is non-zero
        dQ = np.zeros_like(Q)
        dQ[np.arange(batch), actions] = weights * huber_grad / batch

        # --- Dueling aggregation backward ---
        # Q = V + A - mean(A)  ⟹  dV = sum(dQ), dA = dQ - mean(dQ)
        dV = dQ.sum(axis=1, keepdims=True)               # (batch, 1)
        dA = dQ - dQ.mean(axis=1, keepdims=True)         # (batch, action_dim)

        # --- Advantage stream backward (a1: hidden→64, a2: 64→action_dim) ---
        # Layer a2: (batch, 64) @ (64, action_dim)  → dA is (batch, action_dim)
        daa1 = dA @ self.a2.W.T                          # (batch, 64)
        dza1 = daa1 * _relu_grad(self._cache["aa1"])     # (batch, 64)
        dA1_W = self._cache["aa1"].T @ dA                # (64, action_dim)
        dA1_b = dA.sum(0)                                # (action_dim,)
        # Propagate through a1 (hidden → 64): W shape (hidden, 64)
        dA0_W = self._cache["a2"].T @ dza1               # (hidden, 64)
        dA0_b = dza1.sum(0)                              # (64,)
        d_a2_from_adv = dza1 @ self.a1.W.T              # (batch, hidden)

        # --- Value stream backward (v1: hidden→64, v2: 64→1) ---
        # Forward: zv1 = a2_shared @ v1.W + v1.b  shape (batch, 64)
        #          av1 = relu(zv1)
        #          V   = av1 @ v2.W + v2.b          shape (batch, 1)
        # Backward:
        dav1 = dV @ self.v2.W.T                          # (batch, 64)
        dzv1 = dav1 * _relu_grad(self._cache["zv1"])    # (batch, 64)
        dV1_W = self._cache["av1"].T @ dV               # (64, 1)
        dV1_b = dV.sum(0).squeeze()                      # (1,) → scalar-safe
        dV0_W = self._cache["a2"].T @ dzv1              # (hidden, 64)
        dV0_b = dzv1.sum(0)                              # (64,)
        d_a2_from_val = dzv1 @ self.v1.W.T              # (batch, hidden)

        # --- Shared trunk backward ---
        d_a2 = d_a2_from_adv + d_a2_from_val            # (batch, hidden)
        d_z2 = d_a2 * _relu_grad(self._cache["z2"])     # (batch, hidden)
        dfc2_W = self._cache["a1"].T @ d_z2             # (hidden, hidden)
        dfc2_b = d_z2.sum(0)                            # (hidden,)
        d_a1 = d_z2 @ self.fc2.W.T                      # (batch, hidden)
        d_z1 = d_a1 * _relu_grad(self._cache["z1"])     # (batch, hidden)
        dfc1_W = s.T @ d_z1                             # (state_dim, hidden)
        dfc1_b = d_z1.sum(0)                            # (hidden,)

        # --- Global gradient-norm clip ---
        all_grads = [
            dfc1_W, dfc1_b, dfc2_W, dfc2_b,
            dV0_W, dV0_b, dV1_W, dV1_b,
            dA0_W, dA0_b, dA1_W, dA1_b,
        ]
        gnorm = np.sqrt(sum(float(np.sum(g ** 2)) for g in all_grads))
        if gnorm > self.clip_norm:
            scale = self.clip_norm / gnorm
            all_grads = [g * scale for g in all_grads]
        (dfc1_W, dfc1_b, dfc2_W, dfc2_b,
         dV0_W, dV0_b, dV1_W, dV1_b,
         dA0_W, dA0_b, dA1_W, dA1_b) = all_grads

        # --- Adam updates ---
        self._t += 1
        kw = dict(lr=self.lr, beta1=self.BETA1, beta2=self.BETA2,
                  eps=self.EPS_ADAM, t=self._t)
        self.fc1.update(dfc1_W, dfc1_b, **kw)
        self.fc2.update(dfc2_W, dfc2_b, **kw)
        self.v1.update(dV0_W, dV0_b, **kw)
        self.v2.update(dV1_W, dV1_b.reshape(-1), **kw)
        self.a1.update(dA0_W, dA0_b, **kw)
        self.a2.update(dA1_W, dA1_b, **kw)

        return loss, td_errs

    # ------------------------------------------------------------------
    # Target-network helpers
    # ------------------------------------------------------------------

    def copy_from(self, other: "DuelingQNetwork") -> None:
        """Hard copy all weights (used at init)."""
        for dst, src in self._param_pairs(other):
            dst.W = src.W.copy()
            dst.b = src.b.copy()

    def soft_update_from(self, other: "DuelingQNetwork", tau: float) -> None:
        """Polyak averaging: self ← tau*other + (1-tau)*self."""
        for dst, src in self._param_pairs(other):
            dst.W = tau * src.W + (1.0 - tau) * dst.W
            dst.b = tau * src.b + (1.0 - tau) * dst.b

    def _param_pairs(self, other: "DuelingQNetwork"):
        return zip(
            [self.fc1, self.fc2, self.v1, self.v2, self.a1, self.a2],
            [other.fc1, other.fc2, other.v1, other.v2, other.a1, other.a2],
        )