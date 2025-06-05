# Reinforcement Learning Projects 

## Overview

This repository contains a collection of projects and implementations developed as part of the Reinforcement Learning (RL) course 
at National Polytechnic University of Armenia (NPUA) during the Spring semester of the 2024/2025 academic year.
The projects focus on applying RL algorithms to solve classic problems, such as training agents to play games or optimizing decision-making in simulated environments.

---

## Repository Structure
The repository is organized into subdirectories, each containing a distinct RL project inspired by 
concepts from [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf) by Richard S. Sutton and Andrew G. Barto.

---

## Projects 

### [Project 1: Tic-Tac-Toe](tic-tac-toe/)
- **Description**: Train an RL agent to play Tic-Tac-Toe, with features like early tie detection and human-play mode.
- **Main File**: [tic_tac_toe.py](tic-tac-toe/src/tic_tac_toe.py)

### [Project 2: Ten-armed-Testbed](ten-armed-testbed/)
- **Description**: Train an RL agent (`Bandit`) to select the best action. Demonstrate multiple action selection strategies. Maximize the expected reward by selecting the best actions based on learned estimates.
- **Main File**: [ten-armed-testbed.ipynb](ten-armed-testbed/notebooks/ten_armed_testbed.ipynb)

### [Project 3: Grid-World](gridworld-mdp/)
- **Description**: Train an RL agent which will explore the environment and update state values to solve a 5x5 Gridworld using the Bellman optimality equation. Visualize the learned state-value function and optimal policy.
- **Main File**: [grid-world.py](gridworld-mdp/src/grid_world.py)

### [Project 4: Grid World – Dynamic Programming](gridworld-dp/)
- **Description**: Evaluate the state-value function of a grid-based environment using Dynamic Programming. Implements iterative policy evaluation under a uniform random policy in a 4x4 grid with terminal states.
- **Main File**: [grid_world.py](gridworld-dp/src/grid_world.py)

### [Project 5: Gamblers Problem – Value Iteration](gambler-problem/)
- **Description**: Solve the Gambler’s Problem using Value Iteration to find the optimal policy that maximizes the probability of reaching a target capital. Visualizes value function updates and the final betting strategy.
- **Main File**: [gamblers_problem.ipynb](gambler-problem/notebooks/gamblers_problem.ipynb)

### [Project 6: Blackjack – Monte Carlo Methods](blackjack/)  
- **Description**: Estimate state-value and state-action functions in the Blackjack environment using Monte Carlo methods. Includes On-Policy evaluation, Exploring Starts for optimal policy estimation, and Off-Policy learning with importance sampling.  
- **Main File**: [black_jack.py](blackjack/src/black_jack.py)

### [Project 7: Infinite Variance – Importance Sampling](infinite-variance/)  
- **Description**: Illustrate the instability of ordinary importance sampling in off-policy evaluation through a simple stochastic environment. Visualizes high-variance estimates as a result of diverging behavior and target policies.  
- **Main File**: [infinite_variance.py](infinite-variance/src/infinite_variance.py)

### [Project 8: Random Walk environment – MC vs TD](random-walk/)
- **Description**: Evaluate state-value functions in a linear random walk environment using both Temporal Difference and Monte Carlo methods. Includes online learning and batch updates, with RMSE-based comparisons against true values.  
- **Main File**: [random_walk.py](random-walk/src/random_walk.py)

### [Project 9: Windy Grid World – SARSA](windy-gridworld/)
- **Description**:  Solve the Windy Grid World using the SARSA algorithm with ε-greedy action selection. The agent learns to navigate a stochastic, wind-affected grid through online TD updates.
- **Main File**: [windy_grid_world.py](windy-gridworld/src/windy_grid_world.py)

### [Project 10: Cliff Walking – SARSA, Expected SARSA, Q-Learning](cliff-walking/)
- **Description**: Solve the Cliff Walking environment using SARSA, Expected SARSA, and Q-learning. Compares safe vs optimal policies, reward trends, and sensitivity to step-size using multiple learning algorithms.  
- **Main File**: [cliff_walking.py](cliff-walking/src/cliff_walking.py)

### [Project 11: Maximization Bias – Q-Learning vs Double Q-Learning, Expected SARSA vs Double Expected SARSA](maximization-bias/)
- **Description**: Compare how standard Q-Learning suffers from maximization bias in a noisy two-state environment, and evaluate how Double Q-Learning mitigates that bias by more accurate action‐value estimation. Compare Expected SARSA and Double Expected SARSA to show that even without max() operator, still the policy can be improved using two different estimates.
- **Main File**: [maximization_bias.py](maximization-bias/src/maximization_bias.py)


---

## Installation

Ensure Python 3.x is installed.

### Clone the repository 
```bash 
git clone https://github.com/ZhasminHovhannisyan/Reinforcement-Learning.git
cd Reinforcement-Learning
```
### Install dependencies
```bash
pip install -r requirements.txt
```
More detailed information about each project can be found in corresponding directories.
Then run the main script for each project.
