# Random Walk - TD(0) vs MC Comparison

## Project Overview
This project implements the Random Walk environment to 
compare two foundational reinforcement learning methods: 
Temporal Difference (TD(0)) and Monte Carlo (MC). 
The environment consists of a linear chain of 7 states, 
where the agent starts in the middle and takes
random steps until reaching a terminal state.

The goal is to estimate the state-value function of 
non-terminal states based on sampled episodes. 
Both online and batch versions of TD(0) and MC are evaluated
and visualized using Root Mean Square Error (RMSE) 
compared to the known true values. 
This setup provides a clear demonstration of how TD and MC 
differ in convergence behavior, sample efficiency, and stability.

---

## Project Files


- **[random_walk.py](src/random_walk.py)**: Core implementation of the environment and learning algorithms.
  - `true_values`: Returns the analytically computed true state values.
  - `approximate_values`: Initializes the state-value function with zeros.
  - `temporal_difference`: Performs a single TD(0) update (with optional step size).
  - `monte_carlo`: Performs a single Monte Carlo update (with optional step size).
  - `batch_updating`: Implements batch training for both TD and MC.

- **[random_walk.ipynb](notebooks/random_walk.ipynb)**: Jupyter notebook for running experiments and plotting results.
  - Plots value estimates after 0, 1, 10, and 100 episodes of TD(0).
  - Compares convergence of TD and MC with various step sizes.
  - Visualizes RMSE over episodes for both **online** and **batch** settings.

- **[requirements.txt]()**: Dependencies for the project
- **[README.md]()**: Project documentation

---


## How it Works

### Environment

The environment is a simple linear chain of 7 states:
Terminal-Left — A — B — C — D — E — Terminal-Right

The agent always starts in the middle state `C` and takes random steps (left or right) until it reaches a 
terminal state. The episode ends when the agent reaches either terminal state. 
The reward is **+1** for reaching the right terminal state, and **0** otherwise.
The goal is to estimate the **value of each non-terminal state** using:

- **TD(0) Learning**
- **Monte Carlo Learning**


The true value of each non-terminal state is its probability of reaching the right terminal.


#### Batch Updating
- Stores all episodes so far and repeatedly updates the value function until convergence.
- Implemented for both TD(0) and Monte Carlo.

---

### Parameters

You can configure hyperparameters in the [notebook](notebooks/random_walk.ipynb):

- step_size = 0.1         ``# Learning rate α`` 
- episodes = 100          ``# Number of episodes``
- runs = 100              ``# Independent runs for averaging``


---

### Learning Methods

### Temporal Difference (TD(0))

Updates value estimates after each step using bootstrapping:

V(s) ← V(s) + α [R + γ V(s') − V(s)]

### Monte Carlo (MC)

#### Monte Carlo
Value estimates are updated at the end of each episode based on total return:

V(s) ← V(s) + α [G − V(s)]

Batch updating is implemented for both TD(0) and MC, which updates values repeatedly over a batch of episodes until convergence.


---

## Visualizations

### 1. **State Value Estimates Over Time (TD Only)**  
Plots the estimates after 0, 1, 10, and 100 episodes compared to true values.

### 2. **RMSE Comparison: TD vs MC**

- Runs multiple episodes and compares RMSE over time.
- Includes multiple learning rates (α) for each method.

### 3. **Batch Updating Comparison**
Shows how RMS error evolves for both methods during batch learning.

---