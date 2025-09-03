# Random Walk: n-step Temporal Difference Learning

## Project Overview
This project implements the **Random Walk** environment to study **n-step Temporal Difference (TD) learning**.  
An agent starts from the center of a 19-state chain and moves left or right with equal probability until reaching one of two terminal states:  
- **Left terminal** → reward = −1  
- **Right terminal** → reward = +1  

The goal is to estimate the **state-value function** for each non-terminal state using **n-step TD methods** and analyze how the parameter \(n\) affects learning performance and bias–variance trade-offs.

---

## Project Files

- **[random_walk.py](src/random_walk.py)**  
  - Defines the Random Walk environment with 19 states.  
  - Implements **n-step TD learning** update rule
  - Runs episodes and updates state-value estimates over time.

- **[random_walk.ipynb](notebooks/random_walk.ipynb)**  
  - Simulates multiple runs of **n-step TD** for different values of \(n\).  
  - Computes root mean-squared error (RMSE) against the **true state-values** (from the Bellman equation).  
  - Produces comparative plots showing how varying \(n\) influences convergence.

- **[requirements.txt](requirements.txt)**  
  - Project dependencies.

- **[README.md](README.md)**  
  - Project documentation.

---

## How it Works

### Environment
- States: 19 non-terminal states in a line, plus two terminals.  
- Start: always in the center state.  
- Transitions: agent moves **left** or **right** with equal probability (0.5 each).  
- Rewards:  
  - −1 if episode terminates on the left,  
  - +1 if it terminates on the right,  
  - 0 otherwise.  

### n-step TD Learning
The update is based on the **n-step return**:

n-step return:

G_t^(n) = Σ (from k=0 to n-1) [ γ^k * R_{t+k+1} ]  +  γ^n * V(S_{t+n})

Value function update:

V(S_t) ← V(S_t) + α * ( G_t^(n) − V(S_t) )

Where:
- \(n\) controls the number of steps into the future considered for updates.  
- Smaller \(n\) → **more biased**, less variance.  
- Larger \(n\) → **less biased**, more variance.


### Parameters
You can configure the computation via:
```python
states_number = 19,                         # Number of states
discount=1.0,                               # Discount factor (γ)
steps = np.power(2, np.arange(0, 10)),      # Number of steps to look ahead
step_sizes = np.arange(0, 1.1, 0.1),        # Step-size parameter (α)
episodes=10,                                # Number of episodes per run
runs=100                                    # Number of independent runs
```
---

## Results & Visualization
The notebook visualizes:
- **[Learning curves](generated_images/)** for different values of \(n\).  
- **RMSE vs. episodes** to measure accuracy compared to true state-values.  
- Comparison showing the **bias–variance trade-off** inherent in n-step TD methods.

---
