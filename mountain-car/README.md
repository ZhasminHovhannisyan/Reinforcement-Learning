# Mountain Car – Semi‑Gradient n‑Step SARSA with Tile Coding

## Project Overview

This project implements semi‑gradient n‑step SARSA to solve the classic Mountain Car reinforcement‑learning problem using tile coding for function approximation.  
The objective is to train an underpowered car to climb a steep hill by learning value estimates that encourage momentum‑building swings.

The goal of this project is to demonstrate value function approximation in continuous state spaces, specifically the following techniques:

- Tile‑coded value function approximation
- Semi‑gradient n‑step SARSA
- Continuous control (position & velocity)
- Cost‑to‑go visualization

---

## Project Files

- **[mountain_car.py](src/mountain_car.py)**: Mountain Car environment & learning loop
  - Continuous state space (position, velocity)
  - Actions: reverse (−1), zero (0), forward (+1)
  - Reward = −1 each step until reaching goal at position 0.5
  - `ValueFunction` with tile coding
  - `semi_gradient_n_step_sarsa()` learning loop
  - 3D cost‑to‑go visualization

- **[tile_coding.py](/src/tile_coding.py)**: Tile‑coding utility module:
  - `IHT` (Index Hash Table) for collision management
  - `tiles()` for mapping state‑action features to sparse indices

- **[mountain_car.ipynb](notebooks/mountain_car.ipynb)**: Jupyter notebook to visualize results:
  - Learning runs
  - Cost‑to‑go surface plots
  - Effect of n‑step parameter

- **[requirements.txt](requirements.txt)**: Project dependencies
- **[README.md](README.md)**: Project documentation

---

## How It Works

### Environment
- Position ∈ [−1.2, 0.5]
- Velocity ∈ [−0.07, 0.07]
- Episode terminates when position = 0.5
- Reward = −1 per step

### Dynamics
Velocity update:
```
v ← v + 0.001 * action − 0.0025 * cos(3 * position)
v clipped to [−0.07, 0.07]
```

Position update:
```
p ← p + v
p clipped to [−1.2, 0.5]
```

### Algorithm (n‑step SARSA target)
For each episode:

1. Start at a random position near the valley bottom, zero velocity

2. Select an action using greedy policy (ε = 0 with optimistic initialization)

3. Collect rewards and transitions for n steps

4. Compute n-step return:

```
G = r1 + r2 + ... + rn + V(s_n, a_n)
```
5. Update weights using semi-gradient:
```
w <- w + α (G − V(s, a)) ∇V(s, a)
```

### Tile Coding
State variables are scaled and fed to overlapping discrete tilings to produce sparse binary features for linear value estimation.

The state-action features are tile-coded:

- Multiple overlapping tilings (default: 8)

- Each tile activates one weight index

- Scaled inputs ensure smoother generalization:

```
position_scale = n_tilings / (pos_max − pos_min)
velocity_scale = n_tilings / (vel_max − vel_min)
```

Value Estimate
```
Q_hat(s,a) = Σ weights[active_tiles(s,a)]
```

---

### Parameters
```
num_tilings = 8
max_size = 2048       # tile table size
step_size = 0.5       # α distributed across tilings
n_steps   = 4         # n-step SARSA
episodes  = 500
```
---

## Results & Visualization

The [notebook](/notebooks/mountain_car.ipynb) demonstrates:

- Cost-to-go surface across training episodes

- Car's learning curve, which shows that fewer steps are needed to reach the goal over time

- Effect of n-step returns on stability and convergence

Typical learning behavior in early episodes demonstrates, that the car oscillates many times before reaching the hilltop.
However, in the later episodes, when the car learned momentum coordination, reaches goal efficiently

---

## Summary

Tile coding + semi‑gradient n‑step SARSA provides an efficient method for solving continuous‑state RL tasks.  
Mountain Car remains a foundational benchmark for studying function approximation and control in reinforcement learning.
