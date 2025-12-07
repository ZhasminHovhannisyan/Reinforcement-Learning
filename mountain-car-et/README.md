# Mountain Car – SARSA(λ) with Eligibility Traces and Tile Coding

## Project Overview

This project implements **SARSA(λ)** with several **eligibility trace mechanisms** to solve the continuous-state **Mountain Car** task.

The purpose is to compare how different trace implementations affect bootstrapping, control performance, and convergence speed when combined with **tile-coded function approximation**.

---

## Project Files

- **[mountain_car.py](src/mountain_car.py)**: The main implementation file containing:
  - Mountain Car environment and transition dynamics  
  - Action selection  
  - SARSA(λ) implementation  
  - Eligibility trace update rules  
  - Play & evaluation loop  

- **[tile_coding.py](src/tile_coding.py)**  
  - Index Hash Table  
  - Tile coding utilities
  - Feature extraction for continuous states

- **[mountain_car.ipynb](notebooks/mountain_car.ipynb)**
  - Training comparison of eligibility-trace methods  
  - Learning curves  
  - Episode-length reduction  
  - Cost-to-go visualization  

- **[requirements.txt](requirements.txt)**: Project dependencies.

- **[README.md](README.md)**: Project documentation.

---

## How It Works

### Environment
- State: `(position, velocity)`
- Action space: reverse (−1), neutral (0), forward (+1)
- Reward: −1 until reaching goal position
- Episode ends when position ≥ 0.5
- Continuous dynamics

### Transition equations
v = v + 0.001a − 0.0025cos(3p)
p = p + v
p ∈ [-1.2, 0.5]
v ∈ [-0.07, 0.07]


---

## SARSA(λ) with Tile Coding

### Step sequence
1. observe state `(position, velocity)`
2. choose action using ε-greedy
3. apply action + get reward
4. choose next action
5. update eligibility accumulator
6. weight update via TD error

### Eligibility Trace Types
- Accumulating trace  
- Dutch trace  
- Replacing trace  
- Replacing trace with clearing (action-exclusive)

TD Error

delta = reward + V(next_state, next_action) − V(state, action)


Value approximation

Q(s,a) = sum of weights over active tiles(s,a)


---

## Parameters
```
num_tilings = 8
max_size = 2048
step_size = 0.5
lambda = 0.0 … 1.0
episodes = 500
trace = accumulating | dutch | replacing | clearing
```

---

## Results & Visualization

The notebook demonstrates:
- **episode length over training** (faster convergence for Dutch and replacing traces)
- **comparison across λ values**
- **evaluating action-value estimates during training**
- **cost-to-go surface** illustrating improved control over time

Visualization includes:
- learning curves showing reduction in steps-to-goal,
- side-by-side comparison of trace variants,
- 3D surface plots of estimated cost-to-go.

Typical behavior: early episodes take thousands of steps due to oscillation; later, the agent learns to swing efficiently and reaches the goal rapidly.

---

## Summary

This project shows that:
- eligibility traces accelerate learning compared to simple TD(0),
- replacing and Dutch traces improve credit assignment,
- tile coding effectively handles continuous state variables.

Mountain Car remains a standard benchmark for demonstrating value-function approximation and the practical impact of SARSA(λ) methods.
