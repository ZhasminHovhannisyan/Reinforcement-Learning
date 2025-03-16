# k-Armed Bandit Problem Implementation

## Project overview
This project implements the K-Armed Bandit problem using Python. The implementation supports different action selection strategies, including:
- **ε-greedy algorithm**
- **Upper-Confidence-Bound (UCB) selection**
- **Gradient Bandit Algorithm (GBA)**
- **Optimistic Initial Values method**

The goal of the k-armed bandit problem is to maximize the expected reward by selecting the best actions based on learned estimates.



## Project Files

- **[bandit.py](src/bandit.py)**: Implements `Bandit` class, which demonstrates k-armed bandit problem. Includes different action selection strategies, reward calculations, and learning mechanisms.
  * Initializes bandit arms with expected reward values.
  * Supports ε-greedy, UCB, and gradient-based selection strategies.
  * Implements incremental updates for efficient learning.
  * Tracks optimal action selection frequency.

- **[ten_armed_testbed.ipynb](notebooks/ten_armed_testbed.ipynb)**: Provides an interactive testbed for experimenting with different bandit algorithms.
  * Simulates multiple bandit runs to compare learning strategies.
  * Visualizes cumulative rewards and optimal action rates over time.
  * Allows customization of parameters to analyze different exploration-exploitation trade-offs.

- **[requirements.txt](requirements.txt)**: Dependencies for the project
- **[README.md](README.md)**: Project documentation



## How it works

### Configuration
The `Bandit` class provides several options for customizing the learning process:
```python
bandit = Bandit(
    arms_number=10,           # Number of arms (k)
    use_sample_averages=True, # Use sample-average method
    epsilon=0.1,              # Exploration probability for ε-greedy
    confidence_level=2,       # UCB confidence level (if using UCB)
    use_gradient=True,        # Use gradient bandit algorithm
    step_size=0.1,            # Step size for updates
    use_gradient_baseline=True, # Use baseline for gradient method
    initial_action_value_estimates=5.0 # Optimistic initial estimates
)
```


### Action Selection Methods
- **ε-greedy:** With probability ε, a random action is selected; otherwise, the best-known action is chosen.
- **Upper-Confidence-Bound (UCB):** Selects actions based on an optimistic estimate that includes a confidence interval.
- **Gradient Bandit Algorithm (GBA):** Uses policy gradient methods to learn probabilities for each action.
- **Optimistic Initial Values:** Encourages initial exploration by starting with high estimated action values.


## Results & Visualization
The Jupyter notebook provides plots to compare the performance of different strategies 
based on cumulative rewards and optimal action selection percentages.