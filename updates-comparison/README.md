# Updates Comparison – Expected vs Sample Updates

## Project Overview

This project explores the **trade-off between expected and sample updates** in value estimation.
It demonstrates how the computational effort required for each type of update affects the **accuracy** and **efficiency** of learning, particularly in environments with large branching factors.

In essence, the project answers the question:  
> “Given limited computational resources, is it better to perform one expensive expected update, or many cheaper sample updates?”

---

## Project Files

- **[expectation_vs_sample.py](src/expectation_vs_sample.py)**  
  - Implements the `calculate_errors(branching_factor)` function to estimate the RMSE in value prediction for different branching factors.  
  - Models the process of repeatedly sampling from a value distribution and compares the convergence speed toward the true value.

- **[expectation_vs_sample.ipynb](notebooks/expectation_vs_sample.ipynb)**  
  - Reproduces the **Expected vs. Sample Update** analysis from the textbook.  
  - Simulates environments with branching factors `b ∈ {2, 10, 100, 1000}`.  
  - Plots RMS error over computation time, normalized by `b`, to illustrate efficiency gains from sampling-based updates.  

- **[requirements.txt](requirements.txt)**  
  - Project dependencies.

- **[README.md](README.md)**  
  - Project documentation.
---

## How It Works

### Expected vs Sample Updates

For a given state–action pair (s, a):
- Let **b** denote the *branching factor* — the number of possible next states s′ with nonzero transition probability p(s′|s,a) > 0.
- An **expected update** computes the full expectation over all next states:
  Q(s,a) ← ∑ₛ′ p(s′|s,a)[r + γ maxₐ′ Q(s′,a′)]
  which requires evaluating **b** terms.
- A **sample update** instead draws one next state s′ from the distribution and performs:
  Q(s,a) ← r + γ maxₐ′ Q(s′,a′)
  which is faster but introduces sampling error.

---

### Algorithmic Simulation

Each experiment performs the following steps:

1. **Generate Value Distribution**  
   Randomly sample `b` next-state values from a normal distribution N(0, 1).

2. **Compute True Value**  
   Compute the exact mean — representing the *expected update* result.

3. **Iteratively Sample Updates**  
   Draw random samples (simulating sample-based updates) and compute the absolute error between the sample mean and the true value.

4. **Repeat and Average**  
   Perform multiple runs (e.g. 100) to average errors across trials and obtain smooth convergence curves.

---

### Parameters

Configurable in [`expectation_vs_sample.py`](src/expectation_vs_sample.py):

```python
branching_factors = [2, 10, 100, 1000]  # Number of possible next states (b)
runs = 100                               # Number of independent runs
samples_per_run = 2 * branching_factor   # Number of updates per run
```

---

## Results & Visualization

The [notebook](notebooks/expectation_vs_sample.ipynb) plots 
the comparison [Expected vs Sample Update Efficiency](generated_images/figure_8_7.png), where
- X-axis: Number of computations normalized by branching factor (0 → 2b).  
- Y-axis: RMS error in value estimate.

**Observations:**

For small branching factors, expected updates provide clear accuracy advantages. 

For large **b**, a few sample updates achieve almost the same error reduction as a full expected update, with much less computation. 

This demonstrates that **sample-based learning** scales more efficiently for large problems, making it the preferred approach in most RL applications.

---

## Summary

This project highlights a fundamental insight in reinforcement learning:  
- **Expected updates** are accurate but computationally expensive.  
- **Sample updates** are noisy but computationally cheap and widely applicable.

In large environments with many state–action pairs, **sample updates dominate** in practice — a principle that underpins algorithms like **Q-learning** and **SARSA**.
