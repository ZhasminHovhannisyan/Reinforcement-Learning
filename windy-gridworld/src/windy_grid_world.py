import numpy as np

# region Hyper-parameters

# Grid world's height and width
world = dict(height = 7, width = 10)

# Start state coordinates
start = [3, 0]

# Goal state coordinates
goal = [3, 7]

# Possible actions (up, down, left, right)
actions = [0, 1, 2, 3]

# Wind strength for each column
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Discount rate (denoted as 𝛾)
discount = 1

# Reward for each step
reward = -1.0

# Exploration probability (denoted as 𝜀)
exploration_probability = 0.1

# Step-size parameter (denoted as 𝛼)
step_size = 0.5

# endregion Hyper-parameters

# region Functions

def step(state, action):
    # region Summary
    """
    Steps from current state to the next state
    :param state: Current state
    :param action: Action
    :return: Next state
    """
    # endregion Summary

    # region Body

    # Get the state's coordinates
    i, j = state

    # Check if action is "up"
    if action == actions[0]:
        next_state = [max(i - 1 - wind[j], 0), j]

    # Check if action is "down"
    elif action == actions[1]:
        next_state = [max(min(i + 1 - wind[j], world["height"] - 1), 0), j]

    # Check if action is "left"
    elif action == actions[2]:
        next_state = [max(i - wind[j], 0), max(j - 1, 0)]

    # Check if action is "right"
    elif action == actions[3]:
        next_state = [max(i - wind[j], 0), min(j + 1, world["width"] - 1)]

    else:
        assert False

    return next_state

    # endregion Body

def choose_action(action_value_estimates, state):
    # region Summary
    """
    Chooses an action based on 𝜀-greedy algorithm
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param state: State
    :return: Action
    """
    # endregion Summary

    # region Body

    # ε-greedy action selection: every once in a while, with small probability ε, select randomly from among all the actions with equal probability, independently of the action-value estimates.
    if np.random.binomial(n=1, p=exploration_probability) == 1:
        action = np.random.choice(actions)

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    else:
        values = action_value_estimates[state[0], state[1], :]
        action = np.random.choice([act for act, val in enumerate(values) if val == np.max(values)])

    return action

    # endregion Body

def play(action_value_estimates):
    # region Summary
    """
    Play for an episode
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :return: Time steps in episode
    """
    # endregion Summary

    # region Body

    # Track the total time steps in this episode
    time_steps = 0

    # Initialize state at the start
    state = start

    # Choose an action based on 𝜀-greedy algorithm
    action = choose_action(action_value_estimates, state)

    # Keep going until getting to the goal state
    while state != goal:

        # get the next state
        next_state = step(state, action)

        # choose the next action
        next_action = choose_action(action_value_estimates, next_state)

        # SARSA update (Equation (6.7))
        action_value_estimates[state[0], state[1], action] += (
            step_size*(reward + discount * action_value_estimates[next_state[0], next_state[1], next_action]
                       - action_value_estimates[state[0], state[1], action])
        )

        # move to the next state
        state = next_state

        # move to the next action
        action = next_action

        # increment time steps
        time_steps += 1


    return time_steps

    # endregion Body

# endregion Functions
