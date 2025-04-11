# Gambler’s Problem: Value Iteration with Optimal Policy

## Project Overview  
This project implements the **Gambler’s Problem** using **Value Iteration**.  
The gambler makes bets trying to reach a capital of 100, with a biased coin determining wins and losses. The objective is to find the **optimal policy** that maximizes the probability of reaching the goal.

The environment assumes a probabilistic outcome for each bet, and the project visualizes:
- The **value function** over multiple sweeps
- The **final optimal policy**

---

## Project Files

- **[gamblers_problem.ipynb](notebooks/gamblers_problem.ipynb)**: Core implementation of the Gambler’s Problem with value iteration.
  - Defines all states and possible actions (stakes).
  - Implements value updates using the Bellman optimality equation.
  - Tracks history of value functions over sweeps for visualization.
  - Extracts final policy based on optimal action selection.

  - Generates and saves visualizations of:
    - Value estimates across sweeps
    - Final optimal policy (stake per capital)

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

---

## How it Works

### Value Iteration

The algorithm iteratively updates the value function using the Bellman optimality equation:

v(s) ← maxₐ[ p_h × v(s + a) + (1 - p_h) × v(s - a) ]


Where:
- `s` is the current capital
- `a` is the amount staked
- `p_h` is the probability of the coin coming up heads
- `v(0) = 0` and `v(100) = 1` (terminal states)

The optimal policy selects the stake that maximizes the expected return at each capital.

### Parameters

You can adjust the main parameters in the script:
```python
goal = 100              # Final goal capital
head_probability = 0.4  # Biased coin (p_h)
estimation_accuracy = 1e-9  # Convergence threshold
