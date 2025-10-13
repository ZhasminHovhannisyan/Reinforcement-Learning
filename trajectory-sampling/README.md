# Trajectory Sampling – Expected vs On-Policy Updates

## Project Overview

This project studies how the distribution of updates affects learning speed and policy quality in planning with expected one-step updates.
It compares uniform sampling (updates distributed evenly across all state–action pairs) and on-policy sampling (updates weighted toward those actually visited by the ε-greedy policy).

The experiments highlight that while on-policy sampling improves learning
speed early on, uniform sampling achieves better long-term accuracy
This illustrates the trade-off between focus and coverage in planning-based RL.

---

## Project Files

- [trajectory_sampling.py](src/trajectory_sampling.py): Implements the full experimental setup:
  - Defines the Task environment with random transitions and rewards.
  - Provides functions for:
    - `uniform_sampling()` – uniform expected updates across all state–action pairs.

    - `on_policy_sampling()` – updates according to the on-policy visitation distribution.

    - `evaluate_policy()` – Monte Carlo evaluation of the greedy policy.

    - `run_experiment()` – runs both sampling methods over multiple random tasks using multiprocessing.

  - Includes configurable hyperparameters such as branching factor, exploration rate, and max update steps.

- [trajectory_sampling.ipynb](notebooks/trajectory_sampling.ipynb): Runs experiments and visualizes learning performance:

  - Plots value of the start state under the greedy policy vs. computation time (expected updates).

  - Compares results across branching factors (1, 3, 10) and state sizes (1,000 and 10,000).

  - Demonstrates that on-policy updates lead to faster early improvement, but uniform updates catch up and surpass later.

- **[requirements.txt](requirements.txt)**  
  - Project dependencies.

- **[README.md](README.md)**  
  - Project documentation.
## How it Works
### Environment Setup

Each experiment defines a randomly generated episodic MDP:

∣
𝑆
∣ non-terminal states and 2 actions per state.

Each action leads to one of 
𝑏 next states (branching factor).

Each transition has a 0.1 probability of termination.

Rewards are drawn from 
𝑁
(
0
,
1
)
.

Sampling Methods
Uniform Sampling

Cyclically updates every state–action pair:

estimates[s,a] ← (1 − p_term) · E[r + max_a' Q(s', a')]


Ensures even coverage of the environment, suitable for steady long-term convergence.

On-Policy Sampling

Simulates episodes under an ε-greedy policy (ε=0.1) and updates only visited pairs:

a ← ε-greedy(Q(s))

estimates[s,a] ← (1 − p_term) · E[r + max_a' Q(s', a')]


Focuses computation on likely trajectories, accelerating early learning.

## Parameters

Configurable via the top of trajectory_sampling.py:
```
actions = [0, 1]                  # Two possible actions
termination_probability = 0.1     # Episode termination probability
max_steps = 20000                 # Total number of updates
exploration_probability = 0.1     # ε for on-policy sampling

```
Experiment-level parameters in the notebook:
```
num_states = [1000, 10000]        # Number of states
branch_factors = [1, 3, 10]       # Branching factors
methods = [on_policy_sampling, uniform_sampling]
```

## Results & Visualization

The [notebook](notebooks/trajectory_sampling.ipynb) visualizes policy quality vs. computation time.

**Computation–Performance Trade-Off**

- Y-axis: Estimated value of the start state under the greedy policy.

- X-axis: Computation time (in number of expected updates).

- Lines: Different branching factors and sampling strategies.

**Key insights:**

* On-policy sampling learns faster initially by focusing updates on relevant regions.
* Uniform sampling eventually achieves higher accuracy due to broader state–action coverage.
* The effect is stronger with smaller branching factors or larger state spaces.

## Summary

This project demonstrates how the update distribution affects convergence efficiency in planning.
While on-policy focus improves short-term progress, uniform sampling ensures stable long-term value estimates — emphasizing the importance of balancing efficiency and completeness in reinforcement learning updates.