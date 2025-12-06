# Random Walk – Eligibility Traces and λ-Return Methods

## Project Overview

This project investigates different temporal-difference learning algorithms 
for value prediction in the classic 19-state Random Walk task.
The environment consists of a one-dimensional chain of states with terminal rewards 
at both ends. The goal is to estimate state values using different variants 
of **λ-based TD learning**, comparing their learning characteristics and sensitivity to parameters.

Included methods `Off-line, (TD(λ), True Online λ-Return` illustrate key concepts in reinforcement learning:  
bootstrapping, eligibility traces, λ-returns, and incremental learning.

---

## Project Files

- **[random_walk.py](src/random_walk.py)**: Core implementation of the environment and learning algorithms.
    - `OffLineLambdaReturn`: Retrospectively computes λ-returns after each episode.
    - `TemporalDifferenceLambda` Implements accumulating eligibility traces for online learning.
    - `OnLineLambdaReturn` : True Online TD(λ) with Dutch traces for improved stability.
  - Training and RMSE measurement. 
  - Parameter sweep utilities for λ vs α comparison

- **[random_walk.ipynb](notebooks/random_walk.ipynb)**: Jupyter notebook for running experiments and visualizing results. Demonstrates:
  - RMSE performance plots
  - Effect of step size α
  - Effect of λ from 0 → 1 
  for each method respectively.

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

---

## How It Works

### Environment

The environment is a simple linear chain of 19 states:
Terminal-Left(0) — 1 — 2 — ... — 10 — 18 — 19 — Terminal-Right(20). 
Start is always at state 10. 
At each step the agent moves randomly. Moving left or right is chosen with equal probability until reaching a terminal state. 
The agent gets a reward of −1 when reaching state 0, +1 when reaching state 20, and 0 otherwise

True values are analytically derived from symmetry and Bellman equation.


---

## Algorithms

### 1) Off-line λ-Return

At the end of each episode:

- The entire trajectory is stored
- λ-return is computed retrospectively
- Each state is updated once per episode using the λ-return
- Uses weighted combination of n-step returns

Equivalent to Monte-Carlo when λ = 1  
Equivalent to TD(0) when λ = 0  


### 2) TD(λ) — accumulating eligibility traces

Core mechanism is the eligibility trace vector `z`:
z *= λ
z[s] += 1
δ = r + V(s_next) - V(s)
w += α * δ * z

Learning happens *online* at every timestep.


### 3) True Online TD(λ)
Improves TD-lambda by avoiding drift in eligibility traces.
Uses **Dutch traces** and corrected weight updates:

z *= λ
z[s] += 1 - α * λ * z[s]
δ = r + V(s_next) - V(s)

w += α * (δ + V(s) - V_old) * z
w[s] -= α * (V(s) - V_old)

This method tends to be more stable and accurate.

---

## Parameters

Example experiment parameters used:

```
states = 19
episodes = 10
runs = 100

λ ∈ {0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0}
α ∈ [0.0 … 1.0] depending on λ
```

During parameter sweeps, RMSE is measured **after each episode**, averaged across runs.

---

## Results & Visualization

The notebook provides how RMSE changes depending on α (for different λ values) for each method explained above.
- [plot 1](generated_images/figure_12_3.png): off-line λ return algorithm
- [plot 2](generated_images/figure_12_6.png): TD(λ) algorithm
- [plot 3](generated_images/figure_12_8.png): true online TD(λ)

Observations:
- Off-line λ-return converges slowest but most stably
- TD(λ) learns fast but can overshoot
- True Online TD(λ) offers best trade-off

---

## Summary

This project gives practical experimental insight into: Eligibility traces, λ-returns, Online vs offline TD learning and bias-variance trade-offs.

It demonstrates that:

- Increasing λ increases reliance on real returns,
- Decreasing λ increases bootstrapping,
- True Online TD(λ) offers superior stability in incremental learning

