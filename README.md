# Rainbow-lite DQN — CartPole-v1

A from-scratch implementation of a **Rainbow-inspired Deep Q-Network** using only NumPy, solving the Gymnasium `CartPole-v1` environment.

> Built for the Business Analytics Club RL lecture series.

---

## Algorithm

This is a **"Rainbow-lite"** agent — a pure-NumPy implementation combining five SOTA DQN improvements first described in [Rainbow (Hessel et al., 2017)](https://arxiv.org/abs/1710.02298):

| Technique | Paper | Benefit |
|---|---|---|
| **Double DQN** | [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461) | Eliminates Q-value overestimation bias |
| **Dueling Network** | [Wang et al., 2015](https://arxiv.org/abs/1511.06581) | Decouples V(s) and A(s,a) for faster learning |
| **Prioritized Experience Replay** | [Schaul et al., 2015](https://arxiv.org/abs/1511.05952) | Replays high-TD-error transitions more often |
| **N-step Returns** | [Sutton, 1988](http://incompleteideas.net/book/the-book-2nd.html) | Propagates reward signal 3× faster |
| **Soft Target Updates** | [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971) | Stable target drift via Polyak averaging |

Additional engineering: **Huber loss**, **Adam optimizer**, **global gradient norm clipping**.

---

## Architecture

```
Input (4) ──► FC(128) ──► ReLU ──► FC(128) ──► ReLU ──┬──► FC(64) ──► ReLU ──► FC(1)     = V(s)
                                                        └──► FC(64) ──► ReLU ──► FC(2)     = A(s,a)

Q(s,a) = V(s) + [ A(s,a) − mean_a(A(s,a)) ]
```

The **dueling aggregation** lets the network separately learn:
- **V(s):** how valuable is *this state* in general?
- **A(s,a):** how much better is *this specific action* vs the average?

---

## How It Works

### Double DQN
Standard DQN uses `max Q_target(s', a)` — the same network selects *and* evaluates the best next action, systematically overestimating values. Double DQN decouples this:
```
best_a = argmax_a Q_online(s', a)     # online net selects
target = r + γ * Q_target(s', best_a) # target net evaluates
```

### Prioritized Experience Replay (PER)
Uses a **SumTree** data structure for O(log n) priority-based sampling. Each transition gets priority `p = |TD error| + ε`. High-error transitions (where the agent is "surprised") are replayed proportionally more. Importance-sampling weights `w = (1/N · 1/P(i))^β` correct for the non-uniform sampling bias, with `β` annealed from 0.4 → 1.0 over training.

### N-step Returns
Instead of bootstrapping after 1 step, accumulate 3-step discounted returns before storing:
```
G = r_t + γ·r_{t+1} + γ²·r_{t+2}   then bootstrap with  γ³·Q(s_{t+3})
```

### Soft Target Update (Polyak Averaging)
After **every** gradient step:
```
θ_target ← τ·θ_online + (1−τ)·θ_target    τ = 0.005
```
Eliminates the periodic Q-value reset caused by hard target copies.

---

## Project Structure

```
RL-project/
├── network.py          # Dueling Q-Network: forward, Huber backward, Adam, soft update
├── replay_buffer.py    # SumTree, N-step buffer, PrioritizedReplayBuffer
├── dqn_agent.py        # RainbowLiteAgent: Double DQN, PER sampling, action selection
├── train.py            # Training loop, logging, learning curve plot
└── pyproject.toml      # Dependencies (gymnasium, numpy, matplotlib)
```

---

## Setup & Run

```bash
# Install dependencies with uv
uv sync

# Train the agent (headless, saves training_curve.png)
uv run python train.py

# Optional flags
uv run python train.py --render        # visualise episodes
uv run python train.py --episodes 800  # run longer
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 5e-4 | Adam |
| Discount γ | 0.99 | |
| ε start / min | 1.0 / 0.001 | Epsilon-greedy |
| ε decay | 0.990 / episode | |
| Batch size | 64 | |
| Buffer capacity | 50,000 | |
| Soft update τ | 0.005 | per gradient step |
| N-step | 3 | |
| PER α | 0.6 | priority exponent |
| PER β start | 0.4 | annealed to 1.0 |
| Hidden units | 128 | shared trunk |
| Huber δ | 1.0 | |
| Grad clip norm | 10.0 | global |
| Warmup steps | 1,000 | before training starts |

---

## Dependencies

- `gymnasium` — CartPole-v1 environment
- `numpy` — all neural network operations (no PyTorch/TensorFlow)
- `matplotlib` — learning curve plotting