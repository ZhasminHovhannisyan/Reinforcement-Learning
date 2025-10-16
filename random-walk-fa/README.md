# Random Walk with Function Approximation

## Project Overview

This project extends the classical **Random Walk** task to study **n-step Temporal Difference (TD) learning** with **function approximation**.  
The goal is to understand how different **feature representations** — polynomial and Fourier bases — affect the stability and accuracy of value estimation.

The agent performs random walks over discrete states and learns value functions using linear function approximation, testing multiple feature types and TD update steps.

---

## Project Files

- **[random_walk_fa.py](src/random_walk.py)**  
  Core implementation of n-step TD learning with function approximation.  
  Includes environment definition, value updates, and support for different basis functions.

- **[bootstrapping.ipynb](notebooks/bootstrapping.ipynb)**  
  Demonstrates how n-step TD learning performs under varying step sizes and bootstrapping degrees.

- **[polynomials_vs_fourier.ipynb](notebooks/polynomials_vs_fourier.ipynb)**  
  Compares **polynomial** and **Fourier** basis functions for approximating the true value function.

- **[state_aggregation.ipynb](notebooks/state_aggregation.ipynb)**  
  Explores **state aggregation** as a coarse form of function approximation.

- **[tile_coding.ipynb]()**
  Implements tile coding for value function approximation.
  Demonstrates how multiple overlapping tilings capture local features in large continuous or discrete spaces.
  Compares performance and stability of tile-coded TD learning against polynomial and Fourier bases.

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

---

## How It Works

### Environment
- Linear chain of states (e.g., 19 states) with terminal states at both ends.  
- Start always from the center.  
- Each move goes **left** or **right** with equal probability (0.5).  
- Rewards:  
  - −1 if reaching the left terminal.  
  - +1 if reaching the right terminal.  
  - 0 otherwise.

### n-step TD Learning
The update is based on the n-step return:

G_t^(n) = Σ (from k=0 to n−1) [ γ^k * R_{t+k+1} ] + γ^n * V(S_{t+n})

Value function update:

V(S_t) ← V(S_t) + α * ( G_t^(n) − V(S_t) )

### Function Approximation
Instead of storing one value per state, the state-value function is approximated as:

V_hat(s, w) = w^T * x(s)

where `x(s)` is the feature vector for state `s`, and `w` are the learned weights.

The weight update rule is:

w ← w + α * (G_t^(n) − V_hat(S_t, w)) * x(S_t)

### Feature Representations
- **Polynomial basis**: Features are polynomial expansions of normalized state indices.  
- **Fourier basis**: Features are cosine transformations, capturing smoother variations.  
- **State aggregation**: Groups nearby states into bins with shared representations.

---

## Parameters

Example configuration:
```python
num_states = 19  
discount = 1.0                  # γ  
n_values = [1, 2, 4, 8, 16]     # n-step returns  
alpha = 0.01                    # step size  
episodes = 10                   # per run  
runs = 100                      # independent runs  
basis = "fourier"               # or "polynomial" / "aggregation"  
order = 3                       # basis function order  
```
---

## Results & Visualization

The provided notebooks illustrate:
- **[Bootstrapping.ipynb](notebooks/bootstrapping.ipynb)**: Effect of `n` on convergence and stability.
- **[Polynomials_vs_Fourier.ipynb](notebooks/polynomials_vs_fourier.ipynb)**: Fourier basis provides smoother, faster convergence.
- **[State_aggregation.ipynb](notebooks/state_aggregation.ipynb)**: Shows how coarse representations reduce accuracy but improve stability.

---

## Summary
This project shows how function approximation enables generalization in TD learning, allowing agents to scale beyond tabular settings.  
It also demonstrates the **bias–variance trade-off** across different basis functions and the importance of **feature selection** for stable learning.
