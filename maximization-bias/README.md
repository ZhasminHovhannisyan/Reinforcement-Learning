# Maximization Bias – Q-Learning vs Double Q-Learning

## Project Overview

This project implements a simple two-state “biased” environment 
to illustrate **maximization bias** in standard Q-learning and 
how **Double Q-learning** can mitigate that bias. The environment consists of:

- **State A**: Two possible actions (`left` or `right`).  
  - `left` transitions immediately to the terminal state (no noise).  
  - `right` transitions to **State B** (no reward at this transition).
- **State B**: Ten possible actions (indexed 0–9), each
transitioning directly to the terminal state with a stochastic 
reward drawn from N(−0.1,1).

The goal is to show that standard Q-learning tends to overestimate action values
in **State B** (due to noisy samples), causing it to choose “right” too often 
in **State A**. In contrast, Double Q-learning splits the updates across two value
functions, reducing overestimation and yielding a more balanced action choice in **State A**.

---

## Project Files

- **[maximization_bias.py](src/maximization_bias.py)**: Core implementation of the environment and learning algorithms.
  - Functions:
    - `choose_action(action_value_estimates, state)`: ε-greedy selection among available actions in the given `state`.
    - `take_action(state, action)`: Returns reward—zero if `state = A`, or a sample from N(−0.1,1) if `state = B`.
    - `q_learning(first_action_value_estimates, second_action_value_estimates=None)`:  
      - If only `first_action_value_estimates` is provided, runs a single episode of **standard Q-learning**, updating Q1(s,a).  
      - If both `first_...` and `second_...` are provided, runs a single episode of **Double Q-learning**, randomly choosing which value function to update at each step and using the other to compute the target.  
      - Returns `left_count`: the number of times “left” was chosen in State A during that episode.

- **[maximization_bias.ipynb](notebooks/maximization_bias.ipynb)**: Jupyter notebook for running experiments and plotting results.  
  - Contains code to run many independent episodes (1000 runs for 300 episodes) and track how often “left” is chosen in **State A** under each algorithm.  
  - Generates plots of the **frequency of choosing “left”** (i.e., avoiding State B) versus the number of episodes or runs, comparing standard Q-learning to Double Q-learning.

- **[requirements.txt](requirements.txt)**: Dependencies for the project

- **[README.md](README.md)**: Project documentation

---

## How it Works

### Environment Dynamics

- **State A** (ID 0): Two actions  
  1. `right (0)` → transitions to **State B** (ID 1), with **no immediate reward**.  
  2. `left (1)` → transitions to **terminal** (ID 2), with **reward 0**.  
- **State B** (ID 1): Ten actions (0–9)  
  - Any action in State B transitions to **terminal** (ID 2), and returns a reward drawn from a normal distribution N(−0.1,1).


Because each sample in State B is noisy (Gaussian noise), the standard Q-learning update  
$$
Q(s,a) \;\leftarrow\; Q(s,a) \;+\; \alpha\,\Bigl[r + \gamma\,\max_{a'} Q(s',a') \;-\; Q(s,a)\Bigr]
$$

tends to **overestimate** the true value of actions in State B. 
This is known as **maximization bias**.

Double Q-learning splits the action-value function into two independent estimates 
\(Q_1\) and \(Q_2\). At each step, it randomly picks one of the two to update, 
using the other to form the target:  

With probability 0.5, update 𝑄1 using 𝑄2 to select the next action
 $$
Q_{1}(s,a) \;\leftarrow\; Q_{1}(s,a) 
\;+\; \alpha\,\Bigl[r \;+\; \gamma\,Q_{2}\bigl(s',\,\arg\max_{a'}Q_{1}(s',a')\bigr) \;-\; Q_{1}(s,a)\Bigr]
$$

With probability 0.5, update 𝑄2 using 𝑄1 to select the next action
 $$
Q_{2}(s,a) \;\leftarrow\; Q_{2}(s,a) 
\;+\; \alpha\,\Bigl[r \;+\; \gamma\,Q_{1}\bigl(s',\,\arg\max_{a'}Q_{2}(s',a')\bigr) \;-\; Q_{2}(s,a)\Bigr]
$$

This reduces overestimation, so the agent chooses “left” more often in State A 
(avoiding noise in State B).


### Hyperparameters
There is a possibility to try on your custom problem, by changing these parameters
- `states`: Mapping of state names `A`, `B`, and `terminal` to integer IDs.
- `start`: Initial state (`A`).
- `actions_A`: Two actions from State A (`right=0`, `left=1`).
- `actions_B`: Ten actions from State B (`0–9`).
- `state_action_values`: A list of three arrays holding Q-values for A (size 2), B (size 10), and terminal (size 1).  
- `transition`: Defines deterministic next-state mapping:  
  - From `A`: `[terminal, B]`  
  - From `B`: always to `terminal`  
- `exploration_probability` epsilon: 0.1
- `step_size` alpha: 0.1
- `discount` gamma: 1.0  

## Results & Visualization

The [notebook](notebooks/maximization_bias.ipynb) produces two main visualizations in a single plot:

1. **Left‐Action Frequency**  
   - For each algorithm (Q-learning vs Double Q-learning), we run many independent episodes.  
   - Count “how many times the agent picks ‘left’ in State A” in each episode.  
   - Plot the **proportion of “left” choices** (y-axis) versus the **number of episodes** (x-axis).  
   - In early episodes, both algorithms explore and may choose “right” sometimes. Over time:
     - **Q-learning**: tends to overestimate State B’s value, so it picks “right” more often (lower left‐frequency).  
     - **Double Q-learning**: shows a higher left‐frequency, preforming more cautious behavior (less biased).

So, by aggregating over many runs, we plot the asymptotic frequency of “left” for each algorithm.
This shows that **Double Q-learning** converges to a near‐optimal policy that picks “left” almost always, 
while **Q-learning** converges to a suboptimal policy with some “right” bias.

