import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


# region Fields

# Size of rectangular (square) gridworld
grid_size = 4

# Possible 4 actions on a grid (denoted as 𝒜 = {left, up, right, down})
actions = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]

# Suppose the agent selects all 4 actions with equal probability in all states => the probability of each action will be 1/4.
action_probability = 0.25

# endregion Fields

# region Functions

def is_terminal(state):
    # region Summary
    """
    Checks if state is terminal state.
    :param state: State
    :return: True, if state is terminal state; otherwise, False
    """
    # endregion Summary

    # region Body

    # Get the next state's coordinates
    x, y = state

    # Return True if the state is at the vertices of the main diagonal; otherwise, return False
    return (x == 0 and y == 0) or (x == grid_size - 1 and y == grid_size - 1)

    # endregion Body


def step(state, action):
    # region Summary
    """
    Step from current state to next state
    :param state: Current state (denoted as 𝑠)
    :param action: Action taken in current state (denoted as 𝑎)
    :return: Next state (denoted as 𝑠′) and obtained reward (denoted as 𝑟)
    """
    # endregion Summary

    # region Body

    # Check if state is terminal state
    if is_terminal(state):
        return state, 0

    # Next state is obtained by taking an action in the current state
    next_state = (np.array(state) + action).tolist()

    # Get the next state's coordinates
    x, y = next_state

    # Actions that would take the agent off the grid leave its location unchanged
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        next_state = state

    # The reward is -1 on all transitions until the terminal state is reached
    reward = -1

    return next_state, reward

    # endregion Body


def draw(grid):
    # region Summary
    """
    Draw grid of state-value function
    :param grid: State value function grid
    """
    # endregion Summary

    # region Body

    figure, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    width, height = 1.0 / grid.shape[1], 1.0 / grid.shape[0]

    # Add cells
    for (i, j), cell_value in np.ndenumerate(grid):
        table.add_cell(i, j, width, height, text=cell_value, loc='center', facecolor='white')

    # Add external labels for row and column numbers
    for i in range(len(grid)):
        table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none')

    axis.add_table(table)

    # endregion Body


def compute_state_value(in_place=True, discount=1.0, threshold=1e-4):
    # region Summary
    """
    Compute state-value
    :param in_place: True to use 1 array and update the values “in place,” that is, with each new value immediately overwriting the old one; otherwise, False.
    :param discount: Discount rate (denoted as 0 ≤ 𝛾 ≤ 1)
    :param threshold: Small threshold determining accuracy of estimation (denoted as 𝜃 > 0)
    :return: New state-values and number of iterations
    """
    # endregion Summary

    # region Body

    # New values of state-value function table (denoted as 𝑣_𝑘+1 (𝑠))
    new_state_values = np.zeros((grid_size, grid_size))

    # Initialize number of iterations
    iterations = 0

    # Iterate until value convergence
    while True:

        # Compute state-values for in-place and out-of-place cases
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()

        # Old values of state-value function table (denoted as 𝑣_𝑘 (𝑠))
        old_state_values = state_values.copy()

        # Iterate over all states (i.e. on a grid)
        for i in range(grid_size):
            for j in range(grid_size):

                # New state-value
                value = 0

                # For every action
                for action in actions:
                    # get the current state
                    state = [i, j]

                    # get the next state and reward
                    next_state, reward = step(state, action)

                    # compute Bellman equation for 𝑣_𝜋
                    value += action_probability * (reward + discount * state_values[next_state[0], next_state[1]])

                # Assign the computed value as new state-value
                new_state_values[i, j] = value

        # Check value convergence
        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < threshold:
            break

        # Increment number of iterations
        iterations += 1
    return new_state_values, iterations

    # endregion Body

# endregion Functions
