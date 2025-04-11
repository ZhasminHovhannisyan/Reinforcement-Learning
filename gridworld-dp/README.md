# Grid World: Implementation with Dynamic Programming

## Project Overview
This project implements the **Grid World** environment and evaluates
**state-value functions** using **Dynamic Programming** methods. 
It's based on a 4x4 grid world where the agent starts in any non-terminal state
and tries to reach one of the terminal states by randomly selecting
from four possible actions: **Left**, **Right**, **Up**, **Down**.


The environment assumes a uniform random policy, and the goal is to compute the value function for each state under this policy using iterative policy evaluation.

---

## Project Files

- **[grid_world.py](src/grid_world.py)**: Core implementation of the Grid World environment.
  * Defines grid structure and state transitions.
  * Supports terminal state checks, reward assignments, and Bellman updates.
  * Implements iterative value evaluation (both in-place and out-of-place).
  * Includes a visualizer for displaying the value function as a grid.

- **[grid_world.ipynb](notebooks/grid_world.ipynb)**: Interactive notebook for running and visualizing value estimation.
  * Demonstrates how value iteration works.
  * Compares convergence with in-place vs. out-of-place updates.
  * Shows visualizations of the state-value grid across iterations.

- **[requirements.txt](requirements.txt)**: Dependencies for the project
- **[README.md](README.md)**: Project documentation


---

## How it Works

### Value Iteration

The value function is updated iteratively using the Bellman expectation equation:

v(s) ← ∑ₐ π(a|s) ∑_{s',r} p(s',r|s,a) [r + γ v(s')]

In this project:
- The policy is uniform random (π(a|s) = 0.25 for all a).
- The transition probabilities are deterministic.
- Rewards are -1 for each step until the agent reaches a terminal state.

### Parameters
You can configure the computation via:
```bash
compute_state_value(
    in_place=True,       # Use in-place or out-of-place value updates
    discount=1.0,        # Discount factor (γ)
    threshold=1e-4       # Convergence threshold
)
```
---

## Results & Visualization

The `draw()` function displays the final state-value grid after convergence.  
The notebook provides step-by-step iteration results with visual outputs to understand the learning process.
