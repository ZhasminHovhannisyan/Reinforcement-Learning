# Project overview

This project implements a Tic-Tac-Toe game where two RL agents are trained
to play against each other, and a human can compete against a trained RL agent.
The goal is to train the agents in a way so they either win the human player or make a tie.


## Project Files

- **[tic_tac_toe.py](src/tic_tac_toe.py)**: Main script containing the game logic, training, competition, and human-play functions.
- **[state.py](src/state.py)**: Generates all possible Tic-Tac-Toe board states.
- **[player.py](src/player.py)**: Defines `RLPlayer` and `HumanPlayer` classes.
- **[judge.py](src/judge.py)**: Manages the game between two players.


## How It Works

### Training
- Two `RLPlayer` agents are initialized with an exploration rate (`epsilon=0.01`).
- They play against each other for a specified number of epochs.
- After each game, their state-value estimates are updated.
- Policies are saved for later use.

### Competition
- Trained agents compete for a set number of turns.
- Win rates are tracked to assess performance.

### Human Play
- A human player faces a trained RL agent (greedy, `epsilon=0`).
- The RL agent aims to guarantee at least a tie if playing second, assuming optimal play.



## Requirements
- Ensure Python 3.x is installed
- Also numpy library is needed
- After running the tic-tac-toe.py code once, the agents are already trained and the policies are saved, so you can comment train and compete lines
