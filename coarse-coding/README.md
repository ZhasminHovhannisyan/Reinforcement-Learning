# Square Wave Function Approximation – Feature-Based Value Estimation

## Project Overview

This project demonstrates **function approximation** using **feature-based value estimation** and **online gradient descent**.  
The target is a discontinuous **square wave function**, and the model learns to approximate it using a linear value function with overlapping features.  
The implementation showcases how local feature activations and weight updates can represent non-smooth functions in continuous domains.

---

## Project Files

- **[square_wave.py](src/square_wave.py)**  
  Core experiment script:  
  - Defines the **square wave** target function.  
  - Generates random samples from the domain.  
  - Performs training via incremental updates using a value function approximator.

- **[classes.py](src/classes.py)**  
  Implements the supporting classes:  
  - `Interval`: Represents a continuous subrange within the domain.  
  - `ValueFunction`: Linear value function based on **feature windows**.  
    - Each feature covers a small interval.  
    - Multiple features may overlap for smoother representation.  
    - Weights are updated using gradient descent for active features only.  
  - `DOMAIN`: Interval [0, 2) defining the input space.

- **[square_wave.ipynb](notebooks/square_wave.ipynb)**  
  Notebook for training, experimentation, and visualization.  
  - Samples data from the square wave.  
  - Trains a feature-based value function.  
  - Visualizes approximation performance and feature influence.

- **[requirements.txt](requirements.txt)**: Project dependencies  
- **[README.md](README.md)**: Project documentation

---

## How It Works

### Target Function

The **square wave** is defined as:

```
f(x) = 1  if  0.5 < x < 1.5
f(x) = 0  otherwise
```

Samples (x, f(x)) are drawn uniformly from the domain [0, 2).

---

### Value Function Approximation

The approximator divides the domain into overlapping **feature intervals**.  
Each interval (feature) has an associated **weight** wᵢ. For an input x, the estimated value is the sum of all active feature weights:

```
v̂(x) = Σ wᵢ , for i ∈ active(x)
```

Only features that “contain” x are considered active.

---

### Learning Algorithm

For each sample (x, f(x)):

```
δ = f(x) - v̂(x)
wᵢ ← wᵢ + α * δ / n_active
```

where  
- **α** — learning rate (step size)  
- **δ** — prediction error  
- **n_active** — number of active features

This update distributes the correction equally among all active features.

---

## Parameters

Configurable in [`classes.py`](src/classes.py):

```python
feature_width = 0.1          # Width of each feature window
step_size = 0.2              # α: learning rate
num_of_features = 50         # Number of overlapping features
domain = [0.0, 2.0]          # Input space
```

---

## Results & Visualization

The notebook visualizes:  
- The **square wave** target vs. **learned approximation**  
- The effect of **feature width** and **step size** on accuracy  
- How overlapping features allow smooth estimation despite discontinuities

---

## Summary

This project illustrates how **feature-based linear approximators** can learn non-smooth, discontinuous functions through simple gradient updates.  
It serves as a compact demonstration of **function approximation principles** often used in **reinforcement learning** and **value estimation** tasks.
