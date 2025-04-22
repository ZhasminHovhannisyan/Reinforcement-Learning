# Infinite Variance: Off-Policy Evaluation with Importance Sampling

## Project Overview  
This project demonstrates a classic example from reinforcement learning where 
**ordinary importance sampling** suffers from **high variance**. 
It implements a simplified environment to analyze the instability of off-policy evaluation, 
when the behavior and target policies differ significantly.

The environment:
- Terminates with a stochastic reward of 1 or 0 based on the chosen actions.
- Behavior policy chooses between `left` and `right` with equal probability.
- Target policy **always** chooses `left`.

The project visualizes how the **importance sampling ratio** grows exponentially with episode length, 
leading to **infinite variance** in expected return estimates.

---

## Project Files

- **[infinite_variance.py](src/infinite_variance.py)**: Core logic for the simulated environment.
  - Defines target and behavior policies.
  - Simulates episodes based on action transitions.
  - Tracks trajectories for computing importance sampling ratios.

- **[infinite_variance.ipynb](notebooks/infinite_variance.ipynb)**:
  - Runs multiple episodes and simulations.
  - Computes **ordinary importance sampling** estimates.
  - Plots return estimates over episodes (log-scale).

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

---

## How it Works

### Importance Sampling

The **target policy** always selects `"left"`, while the **behavior policy** chooses randomly between `"left"` and `"right"`.

The importance sampling ratio is calculated as:
$$
\rho = \left(\frac{\pi(a \mid s)}{b(a \mid s)}\right)^n = \left(\frac{1}{0.5}\right)^n = 2^n
$$
Where \( n \) is the number of consecutive `"left"` actions before termination.

This exponential growth causes very **high variance** in the estimate of expected return.

### Parameters

You can configure in [jupyter notebook](notebooks/infinite_variance.ipynb):
```python
runs = 10           # Number of independent runs
episodes = 100000   # Number of episodes per run
```

### Results & Visualization
The [final plot](generated_images/figure_5_4.png) shows ordinary importance sampling estimates for each run across episodes.
The x-axis is on a log scale, highlighting how variance increases over time and estimates diverge across runs.
