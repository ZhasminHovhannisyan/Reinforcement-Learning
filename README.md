# Reinforcement Learning Projects 

## Overview

This repository contains a collection of projects and implementations developed as part of Reinforcement Learning (RL) course at NPUA.
The projects focus on applying RL algorithms to solve classic problems, such as training agents to play games or optimizing decision-making in simulated environments.

## Repository Structure
The repository is organized into subdirectories, 
each containing a distinct RL project inspired by 
concepts from *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto.

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
