# Maze Navigation – Dyna-Q, Dyna-Q+, and Prioritized Sweeping

## Project Overview

This project explores model-based reinforcement learning through the **Dyna architecture**, using a maze navigation environment.  
It compares three key algorithms:

- **Dyna-Q** — classical model-based planning approach  
- **Dyna-Q+** — extended version with time-based exploration bonuses  
- **Prioritized Sweeping** — efficient planning focused on high-priority state–action updates  

Each agent learns to reach a goal in a grid maze, with **obstacles**, **changing environments**, and **planning-based updates**.  
The experiments highlight the impact of **model accuracy**, **exploration**, and **planning prioritization** on learning efficiency.

---

## Project Files

- **[maze.py](src/maze.py)**  
  Defines the **Maze** environment:
  - 6×9 grid with start, goal, and obstacle cells.  
  - Supports dynamic obstacle changes and adjustable resolution.  
  - Provides `step()` for transitions and `extend_maze()` for scaling.

- **[models.py](src/models.py)**  
  Contains different **environment models** for Dyna-based planning:
  - `TrivialModel`: deterministic memory of transitions for Dyna-Q  
  - `TimeModel`: adds a *time-based reward bonus* for unvisited states (Dyna-Q+)  
  - `PriorityModel`: implements a *priority queue* and *predecessor tracking* for Prioritized Sweeping  

- **[functions.py](src/functions.py)**  
  Core algorithm implementations:
  - `dyna_q()`: main Dyna-Q and Dyna-Q+ loop with real and simulated updates  
  - `changing_maze()`: tests adaptation when obstacles move mid-training  
  - `prioritized_sweeping()`: model-based learning using a priority queue for efficient backups  
  - `check_path()`: verifies whether learned policy is near-optimal  

- **[dyna.py](src/dyna.py)**  
  Parameter configuration class `DynaParams`, defining:
  - `discount` (γ), `step_size` (α), `exploration_probability` (ε)
  - `time_weight` (κ) for Dyna-Q+
  - `planning_steps`, `threshold` (θ), and number of `runs`

- [dyna_maze.ipynb](notebooks/dyna_maze.ipynb): Run and visualize Dyna-Q vs Dyna-Q+ learning curves
- [changing_maze.ipynb](notebooks/changing_maze.ipynb): Test agent adaptability when the maze changes  
- [prioritized_sweeping.ipynb](notebooks/prioritized_sweeping.ipynb): Demonstrate efficient planning through prioritized updates  

---

## How It Works

### Environment
- Grid world: **6 rows × 9 columns**
- Start: **(2, 0)**  
- Goal: **(0, 8)**  
- Obstacles block certain cells and force detours.
- Reward: +1 for reaching the goal, 0 otherwise.
- Transitions are deterministic.

---

### Algorithms

#### Dyna-Q
Combines real experience with simulated planning updates:

Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]

After each real step, the model generates `n` synthetic experiences for extra updates.

#### Dyna-Q+
Extends Dyna-Q by adding an exploration bonus for actions not tried recently:

r' = r + κ · √(t − τ(s, a))

where `τ(s, a)` is the last time the pair was visited.  
This helps the agent **re-explore** after environmental changes.

#### Prioritized Sweeping
Focuses planning on state–action pairs with **largest TD errors** (high priorities):
1. Store experience in a **priority queue**.  
2. Sample the most urgent pairs first.  
3. Propagate updates backward to predecessors.

---

## Parameters

Configurable through `DynaParams`:

```python
dyna_params = DynaParams()
dyna_params.discount = 0.95          # γ
dyna_params.exploration_probability = 0.1  # ε
dyna_params.step_size = 0.1          # α
dyna_params.time_weight = 1e-4       # κ (Dyna-Q+)
dyna_params.planning_steps = 5       # Number of planning updates per step
dyna_params.threshold = 0.01         # θ (Prioritized Sweeping)
dyna_params.runs = 10                # Number of averaged runs

```

## Results & Visualization
### Dyna-Q vs Dyna-Q+

- Dyna-Q+ achieves faster adaptation to unseen or changed areas due to its exploration bonus.
- Dyna-Q may converge faster initially but can stagnate when the maze changes.

### Changing Maze Experiment

- Obstacles switch position mid-training (obstacle_switch_time).
- Dyna-Q+ quickly re-learns the new optimal path, while Dyna-Q takes longer.

### Prioritized Sweeping

- Demonstrates how selective updates drastically reduce the number of required backups.
- Planning focuses on transitions with the highest impact, improving efficiency.

## Summary
**Dyna-Q** integrates model-based updates but can under-explore.
**Dyna-Q+** uses a simple time-based heuristic to encourage continual exploration.
**Prioritized Sweeping** achieves the same final performance with fewer updates.

Together, these methods illustrate how planning and prioritization accelerate learning in model-based reinforcement learning.