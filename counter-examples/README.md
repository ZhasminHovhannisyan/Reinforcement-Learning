# Baird’s Counterexample – Off-Policy Divergence and Gradient-Corrected Methods

## Project Overview

This project implements Baird’s famous 7-state counterexample — a minimal MDP demonstrating 
that naive off-policy TD learning diverges when used with function approximation. 
The goal is to show that:
* Standard semi-gradient TD does not converge off-policy
* Gradient-TD (GTD/TDC) solves the divergence
* Emphatic TD introduces emphatic weighting for further stability

This experiment addresses one of RL's core problems: stable learning with function approximation in off-policy settings.

## Project Files

- **[counter_example.py](src/counter_example.py)**: Core implementation of Baird's counterexample environment and algorithms.
  - Constructs the Baird 7-state "star" MDP
  - Defines feature representation
  - Implements off-policy semi-gradient TD update
  - Produces classical divergence effect

- **[bairds_counterexample.ipynb](notebooks/bairds_counterexample.ipynb)**
  - Visualizes divergence of TD(0)
  - Shows weight blow-up
  - Illustrates instability from off-policy bootstrapping

- **[tdc_baird.ipynb](notebooks/tdc_baird.ipynb)**
  - Implements GTD/TDC updates
  - Demonstrates stable learning under the same off-policy setting
  - Plots norm of weight vector vs iterations

- **[emphatic_baird.ipynb](notebooks/emphatic_baird.ipynb)**
  - Implements Emphatic TD
  - Shows convergence using emphatic weighting
  - Compares to standard TD and GTD/TDC

- **[requirements.txt](requirements.txt)**: Project dependencies

- **[README.md](README.md)**: Project documentation

## How It Works

### Environment

There are 7 states. States 0–5 are the “upper” states. State 6 is the “lower” state.
This architecture produces directional flow into state 6.

Each state is represented by an 8-dimensional feature vector.
For upper states feature i = 2, last feature = 1. 
For lower state pre-last feature = 1, last feature = 2

This specific representation is critical as it causes classic TD to diverge.

Target policy π always takes solid action, which means it always goes to state 6.
So true value function is: vπ(s) = 0, for all s.

Behaviour policy takes solid with prob 1/7, dashed with prob 6/7.
So sampled experience differs from evaluation target.
This misalignment aims to show the off-policy nature.

**Importance Sampling Ratio**
* for dashed action: ρ = 0
* for solid action: ρ = 1 / (1/7) = 7

## Algorithms
### Semi-gradient TD(0) — OFF-policy
`w ← w + α * ρ * δ * x(s)`

Under Baird’s structure this update **does not follow a gradient**, so TD drifts away from the true value function.
Result: divergence of the weight vector norm.

### Semi-gradient DP (full expectation evaluation)

Unlike TD(0), GTD methods minimize the projected Bellman error using auxiliary weight vectors:
u ← u + β (δ − uᵀφ') φ'
w ← w + α (φ δ − γ φ' (uᵀφ))
Properties:
- **convergent** under off-policy linear approximation
- stable even when TD diverges
- requires auxiliary weights, larger update structure

Result: converges to a fixed weight vector.

### Emphatic TD

Emphatic TD introduces **state-dependent emphatic weights** that correct off-policy distribution mismatch.

Auxiliary weight vector v:

w ← w + α * ρ * (δ x(s) − γ x(s') (vᵀ x(s)))
v ← v + β * ρ * (δ − vᵀ x(s)) * x(s)

Key idea:
- redistribute learning focus according to follow-on trace
- remove instability from off-policy bootstrapping
- convergent without TDC auxiliary weights
- 
Result: converges stably.
Corrects direction of gradient descent.

---

## Parameters 

Typical configuration:
```python
alpha = 0.01        # TD learning rate
beta  = 0.05        # GTD auxiliary rate
gamma = 0.99        # discount factor
episodes = 50000
behaviour_policy = "dashed"
target_policy    = "solid"
```

## Results and Visualizations
The notebooks compare several off-policy TD variants on Baird’s counterexample, showing how different algorithms behave with linear function approximation.

* [plot on top](generated_images/figure_11_2.png) – weight components diverge upward without bound, confirming the classic counterexample result
* [plot on bottom](generated_images/figure_11_2.png) – expected (noise-free) update also diverges, demonstrating that instability is algorithmic, not due to sampling
* [TDC vs Expected-TDC](generated_images/figure_11_5.png) – gradient correction stabilizes learning; both sampled and expected versions converge to a finite solution
* [Expected Emphatic TD](generated_images/figure_11_6.png) – emphatic weighting provides another stable convergent path, approaching a correct fixed point

Observations:
* Semi-gradient off-policy TD diverges in both sample-based and expected forms
* TDC remains stable and convergent thanks to its gradient-correction update
* Emphatic TD converges as well, showing that emphatic weighting is a practical alternative for stable off-policy learning



## Summary

This project illustrates that off-policy TD with function approximation can diverge, 
and the cause is mismatch in update direction due to biased sampling. 
Gradient-TD methods restore stability by following the true gradient. 
Plus, emphatic TD further solves distribution mismatch through weighted importance

So the takeaway from this project is that off-policy learning with function approximation must be done 
with gradient correction or emphatic methods, otherwise it is unstable.