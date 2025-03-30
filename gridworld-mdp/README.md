# Gridworld Value Iteration Implementation

## Project overview
This project implements Value Iteration for solving a Gridworld environment
using RL. The agent explores the environment and updates state values
using the **Bellman Optimality Equation** until convergence.

### Features
- **Gridworld Environment**: A simple 5x5 grid with special states.
- **Value Iteration**: Uses the Bellman Optimality Equation to find the optimal value function.
- **Policy Visualization**: Displays the optimal policy using arrows indicating best actions.
- **Convergence Check**: Iterates until state values stabilize.



## Project Files

- **[grid_world.py](src/grid_world.py)**: Gridworld environment & helper functions
- **[grid_world.ipynb](notebooks/grid_world.ipynb)**: Visualize the learned state-value function and optimal policy
- **[requirements.txt](requirements.txt)**: Dependencies for the project
- **[README.md](README.md)**: Project documentation



## How It Works
### Gridworld Setup
- **Size:** 5x5 grid
- **Special States:**
  - `A → A'` with a reward of **+10**
  - `B → B'` with a reward of **+5**
  - Actions leading out of bounds yield **-1**
- **Possible Actions:** Left (←), Up (↑), Right (→), Down (↓)

### Value Iteration Algorithm
- Initializes state-values to **0**.
- Iteratively updates **state-values** using the Bellman Optimality Equation:
  
  **v(s) = max_a [ R(s,a) + γ * v(s') ]**
  
- Continues until convergence (difference < `1e-4`).
- Extracts the **optimal policy** based on the best action per state.

## Results
**`figure_3_5.png`** - Converged state-value function.
**`figure_3_5_policy.png`** - Optimal policy visualization.
