# Reinforcement Learning Projects 

## Overview

This repository contains a collection of projects and implementations developed as part of the Reinforcement Learning (RL) course 
at National Polytechnic University of Armenia (NPUA) during the Spring semester of the 2024/25, as well as Autumn semester of 2025/26 academic years.
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

### [Project 12: Random Walk – n-step TD Learning](random-walk-ntd/)  
- **Description**: Study the bias–variance trade-off in value estimation using n-step Temporal Difference learning in a 19-state random walk environment. Compares RMSE performance across different step sizes.  
- **Main File**: [random_walk.py](random-walk-ntd/src/random_walk.py)

### [Project 13: Maze Navigation – Dyna-Q, Dyna-Q+, and Prioritized Sweeping](mazes/)
- **Description**: Explore model-based reinforcement learning through maze navigation using Dyna-Q, Dyna-Q+, and Prioritized Sweeping algorithms. Demonstrates how planning, exploration bonuses, and prioritized updates improve learning speed and adaptability in dynamic environments.  
- **Main File**: [functions.py](mazes/src/functions.py)

### [Project 14: Updates Comparison – Expected vs Sample Updates](updates-comparison/)
- **Description**: Analyze the computational trade-off between expected and sample updates in value estimation. Demonstrate how sample-based methods can achieve near-expected accuracy with far less computation, especially in large branching environments. 
- **Main File**: [expectation_vs_sample.py](updates-comparison/src/expectation_vs_sample.py)

### [Project 15: Trajectory Sampling – Expected vs On-Policy Updates](trajectory-sampling/)
- **Description**: Compare uniform and on-policy expected tabular updates in randomly generated MDPs to analyze the effect of update distributions on planning efficiency. Demonstrates how on-policy sampling accelerates early learning but can slow convergence for larger environments.
- **Main File**: [trajectory_sampling.py](trajectory-sampling/src/trajectory_sampling.py)

### [Project 16: Random Walk with Function Approximation – Gradient MC and Semi-Gradient TD](random-walk-fa/)  
- **Description**: Extends the random walk to a 1000-state environment with function approximation using state aggregation, polynomial, Fourier, and tile coding bases. Explores how Gradient MC and Semi-Gradient TD interact with representation quality and bootstrapping.  
- **Main File**: [random_walk_fa.py](random-walk-fa/src/random_walk_fa.py)

### [Project 17: Square Wave – Function Approximation](coarse-coding/)
- **Description**: Implements feature-based value estimation to approximate a discontinuous square wave signal. Uses interval-based feature windows to represent the function and performs incremental weight updates based on sampled data.
- **Main File**: [square_wave.py](coarse-coding/src/square_wave.py)

### [Project 18: Access Control – Differential Semi-Gradient SARSA](access-control/)
- **Description**: Models a continuing decision-making task where a server must accept or reject incoming requests with different priorities. Implements differential semi-gradient SARSA with tile coding, learning average reward and action-values in an off-policy environment. Includes visualizations of long-term server utilization and learned acceptance strategies.
- **Main File**: [access_control.py](access-control/src/access_control.py)

### [Project 19: Mountain Car – Semi-Gradient n-Step SARSA](mountain-car/)
- **Description**: Classic continuous-control environment solved using semi-gradient n-step SARSA with tile-coded state-action features. Demonstrates cost-to-go learning, momentum exploitation, and faster hill-climbing through multi-step bootstrapping.
- **Main File**: [mountain_car.py](mountain-car/src/mountain_car.py)

### [Project 20: Baird’s Counterexample – Off-Policy TD Divergence and Corrective Methods](counter-examples/)
- **Description**: Recreates Baird’s off-policy counterexample to demonstrate divergence of semi-gradient TD. Includes implementation of TDC and Emphatic TD methods, comparing divergence vs. convergence empirically using expected and sample-based updates.
- **Main File**: [counter_example.py](counter-examples/src/counter_example.py)

### [Project 21: Random Walk – Eligibility Traces (TD(λ))](random-walk-et/)
- **Description**: Augments the classical random walk with eligibility-trace methods (offline λ-return, TD(λ), true-online TD(λ)). Compares how λ and bootstrapping affect bias, stability, and RMSE across different return definitions.
- **Main File**: [random_walk_et.py](random-walk-et/src/random_walk.py)

### [Project 22: Mountain Car – SARSA(λ) with Eligibility Traces](mountain-car-et/)
- **Description**: Extends Mountain Car using several eligibility-trace variants including accumulating, Dutch, clearing, and replacing traces. Explores how λ accelerates control learning in continuous states and reduces episode length compared to plain TD.
- **Main File**: [mountain_car_et.py](mountain-car-et/src/mountain_car.py)


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
