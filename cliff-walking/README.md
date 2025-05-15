# Cliff Walking – SARSA, Expected SARSA, and Q-Learning Comparison

## Project Overview

This project implements the **Cliff Walking** environment and compares three reinforcement learning algorithms:
- SARSA (on-policy TD control)
- Expected SARSA
- Q-Learning (off-policy TD control)

The environment features a 4×12 grid with a hazardous "cliff" region. Stepping into the cliff resets the agent to the start state and incurs a large negative reward. The goal is to reach the terminal state with minimal total penalty, while learning the safest or shortest path depending on the algorithm used.



---

## Project Files

- **[`cliff_walking.py`](src/cliff_walking.py)**: Core environment and algorithm implementations.
  - Defines the cliff environment, state transitions, and wind-free dynamics.
  - Implements:
    - `sarsa()` for SARSA
    - `expected=True` in `sarsa()` for Expected SARSA
    - `q_learning()` for Q-learning
  - Uses ε-greedy policy for exploration.
  - Provides `print_optimal_policy()` to visualize learned policies.

- **[`cliff_walking.ipynb`](notebooks/cliff_walking.ipynb)**: Notebook for training, evaluation, and visualization.
  - Compares SARSA vs Q-learning on total episode rewards.
  - Plots optimal policies learned by each algorithm.
  - Evaluates **asymptotic and interim performance** of all three methods across various step sizes.

---

## How it Works

### Environment

- Grid size: 4 rows × 12 columns
- Start: (3, 0), Goal: (3, 11)
- Actions: Up, Down, Left, Right
- The bottom row (cells between start and goal) is the **cliff**: falling into it sends the agent back to start and gives **−100 reward**.

Each step otherwise gives a **−1 reward**, encouraging shorter, safer paths.

---

### Learning Methods

#### SARSA (On-Policy)
Updates values using the next action taken:

Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') − Q(s, a)]

#### Expected SARSA
Uses the expected value over the next action distribution:

Q(s, a) ← Q(s, a) + α [r + γ E_a'[Q(s', a')] − Q(s, a)]


#### Q-Learning (Off-Policy)
Uses the greedy estimate of the next state's best action:

Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]


---

### Parameters

Configurable in [`cliff_walking.py`](src/cliff_walking.py):

```
exploration_probability = 0.1   # ε
step_size = 0.5                 # α
discount = 1                    # γ
```


## Results & Visualization
The [notebook](notebooks/cliff_walking.ipynb) tracks and visualizes the agent’s learning progress across episodes using **SARSA**, **Expected SARSA**, and **Q-learning** algorithms.

### [Episode Reward Comparison](generated_images/example_6_6.png)
- Plots average total rewards per episode for SARSA and Q-Learning over 50 runs.

- Shows how each method balances safe exploration vs optimal (but risky) policies.

### [Step-Size Sensitivity Analysis](generated_images/figure_6_3.png)
- Compares interim (first 100 episodes) and asymptotic (all episodes) performance for:

  - SARSA

  - Expected SARSA

  - Q-learning

- Shows performance trends as the step-size parameter α varies from 0.1 to 1.0.


### Optimal Policy Visualization
Provided in the [notebook](notebooks/cliff_walking.ipynb), this displays learned policies using directional arrows for each cell.

- SARSA tends to learn safer paths that avoid the cliff.

- Q-learning often chooses faster but riskier paths that cut closer to the cliff.
