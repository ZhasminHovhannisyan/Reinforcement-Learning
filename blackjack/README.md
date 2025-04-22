# Blackjack: Monte Carlo Estimation with On/Off-Policy and Exploring Starts

## Project Overview
This project simulates the **Blackjack** card game and applies three **Monte Carlo reinforcement learning** methods:
- **On-Policy Evaluation**: Estimates the state-value function under a fixed player policy.
- **Monte Carlo Exploring Starts (ES)**: Learns optimal state-action values and derives a greedy policy.
- **Off-Policy Evaluation**: Estimates the value of a target policy using episodes from a different behavior policy, with **importance sampling**.


The Blackjack setup simulates a simple card game where the player aims to reach a sum as close to 21 as possible
without going over, while playing against a dealer following a fixed policy. The project estimates:
- **State-value functions**
- **State-action values**
- **Optimal policy estimation**

---

## Project Files

- **[black_jack.py](src/black_jack.py)**: Main implementation of the Blackjack environment and Monte Carlo learning algorithms.
  - Defines game mechanics: player and dealer rules, ace handling, and policies.
  - Implements:
    - `monte_carlo_on_policy()`: On-policy Monte Carlo evaluation.
    - `monte_carlo_es()`: Monte Carlo with Exploring Starts.
    - `monte_carlo_off_policy()`: Off-policy evaluation with both ordinary and weighted importance sampling.
  - Includes helper functions for card drawing, state transitions, and policy behavior.

- **[black_jack.ipynb](notebooks/black_jack.ipynb)**: Interactive notebook to run training and visualize:
  - Value functions for usable vs. non-usable ace states
  - State-action value tables
  - Policy evolution and comparison between Monte Carlo approaches

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

---

## How it Works

### Monte Carlo Learning
The project applies episodic Monte Carlo methods to learn from sampled game episodes, without requiring a model of the environment.

#### On-Policy Estimation
Uses the agent’s target policy (fixed) to evaluate state values based on sample averages from many episodes.

#### Exploring Starts (ES)
Uses random initial states and actions to estimate
optimal state-action values 𝑞(𝑠, 𝑎) and refine a greedy policy

#### Off-Policy Estimation
Estimates the target policy using a different behavior policy with:
- Ordinary importance sampling
- Weighted importance sampling

The behavior policy selects actions randomly; the target policy follows the predefined hitting/sticking strategy.

### Parameters
You can adjust the number of episodes and execution mode in the script:
```python
monte_carlo_on_policy(episodes=500000)
monte_carlo_es(episodes=500000)
monte_carlo_off_policy(episodes=10000)
```
