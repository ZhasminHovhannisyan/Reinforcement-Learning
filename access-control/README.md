# Access Control – Differential Semi‑Gradient SARSA with Tile Coding

## Project Overview

This project implements **differential semi‑gradient SARSA** with **tile coding** for the 
*Access Control Queuing Task*.  
The goal is to learn an optimal policy for a system that decides whether to accept or reject incoming customer requests based on limited server capacity and customer priority, maximizing long‑term average reward.


---

## Project Files

- **[access_control.py](src/access_control.py)**: Main implementation of the access‑control task:
  - Environment dynamics (server usage, customer priority, acceptance decisions)
  - Differential semi‑gradient SARSA algorithm
  - `ValueFunction` class with tile coding for state‑action value estimation
  - ε‑greedy policy with exploration

- **[tile_coding.py](src/tile_coding.py)**: Tile‑coding utilities:
  - `IHT` (Index Hash Table) for managing collisions
  - `tiles()` for feature mapping from continuous state‑action pairs to discrete indices

- **[access_control.ipynb](notebooks/access_control.ipynb)**: Jupyter notebook to run experiments and visualize learning performance.

- **[requirements.txt](requirements.txt)**: Project dependencies.

- **[README.md](README.md)**: Project documentation.

---

## How It Works

### Environment Setup
- **Number of servers**: 10  
- **Priorities**: 0–3  
- **Reward per priority**: 1, 2, 4, 8 (2^priority)  
- **Server release probability**: 0.06 per time step  
- **Actions**:  
  - `accept` (1) → allocate a server (if available)  
  - `reject` (0) → deny service

Each step:
1. A new customer with a random priority arrives.
2. The agent decides whether to accept or reject based on the number of free servers.
3. Servers may become free with probability 0.06.
4. Reward equals `2^priority` if accepted, otherwise 0.

---

### Algorithm (Differential Semi‑Gradient SARSA)

Used for continuing tasks without terminal states.

For each time step:
### **Algorithm Steps (Differential Semi-Gradient SARSA)**

1. **Observe current state:**  
   S_t = (free servers, priority)

2. **Select action** A_t **using ε-greedy policy**

3. **Receive reward** R_{t+1} **and observe next state** S_{t+1}

4. **Select next action:** A_{t+1}

5. **Update weights using the semi-gradient rule:**

   δ = R_{t+1} − R̄ + Q(S_{t+1}, A_{t+1}) − Q(S_t, A_t)

   R̄ ← R̄ + β δ

   w ← w + α δ e_active

**Where:**
- α — step size for state-action value learning  
- β — step size for average reward learning  
- e_active — active tiles from tile coding

---

### Tile Coding Overview

Tile coding enables linear approximation over continuous state spaces.  
Each state–action pair is mapped into a sparse binary feature vector through multiple overlapping tilings.

#### Key Details:
- Number of tilings = configurable (e.g. 8)
- Hash table size = 2048
- Scaled features:
  - `server_scale = n_tilings / num_servers`
  - `priority_scale = n_tilings / (len(priorities) - 1)`

Value estimate:
Q(s,a) = sum w[\text{active tiles}(s,a)]

---

## Parameters

``` python
num_servers = 10
probability_free = 0.06
priorities = [0, 1, 2, 3]
rewards = [1, 2, 4, 8]
step_size_state_action_value = 0.01
step_size_average_reward = 0.01
exploration_probability = 0.1
max_size = 2048
num_tilings = 8
```

---

## Example Usage

``` python
value_function = ValueFunction(num_of_tilings=8)
differential_semi_gradient_sarsa(value_function, max_steps=100000)
```

Typical output:
```
Frequency of number of free servers: [0.03 0.05 0.09 ...]
```

This shows the steady‑state distribution of available servers over time.

---

## Results & Visualization

The [notebook](/notebooks/access_control.ipynb) demonstrates the learning process in the Access Control problem.  
The agent learns how to allocate limited servers optimally based on incoming customer priority.


### 1. Differential Value Plot

- **X-axis:** number of free servers  
- **Y-axis:** differential value of the best action  
- **Colored lines:** different customer priorities (1, 2, 4, 8)

  This graph reveals how the **expected future return** grows with both priority and available capacity, confirming that the agent correctly learned to favor valuable customers when capacity allows.

### Learning Behavior

- The **average reward** stabilizes over time as the system converges to an optimal steady-state policy.  
- The **frequency distribution** of server states (from the training loop) shows that the agent maintains moderate server utilization — balancing acceptance and rejection decisions efficiently.


This indicates that the system operates efficiently around medium load, avoiding both over-rejection and server saturation.


### 2. Priority Heatmap

A heatmap visualizes the **policy learned by the agent**:
- **X-axis:** number of free servers
- **Y-axis:** customer priority (1–8)
- **Cell color:** selected action (accept or reject)

This heatmap clearly shows the policy structure:
- High-priority customers are **almost always accepted**, even when few servers remain.
- Low-priority customers are **mostly rejected** under high load.
  The second visualization shows the **differential state-action values** of the *best action* across server states:

---

## Summary

The **Access Control Task** demonstrates *continuous RL task* using **average‑reward differential SARSA**.  
Combining **tile coding** with **semi‑gradient updates** provides a scalable method to learn efficient resource‑allocation policies in non‑episodic environments.
