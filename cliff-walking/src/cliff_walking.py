import numpy as np

# region Hyper-parameters

# Grid world's height and width
world = dict(height = 4, width = 12)

# Start state coordinates
start = [3, 0]

# Goal state coordinates
goal = [3, 11]

# Possible actions (up, down, left, right)
actions = [0, 1, 2, 3]

# Discount rate for Q-Learning and Expected SARSA (denoted as 𝛾)
discount = 1

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
    :return: Next state and reward
    """
    # endregion Summary

    # region Body

    # Get the state's coordinates
    i, j = state

    # Check if action is "up"
    if action == actions[0]:
        next_state = [max(i - 1, 0), j]

    # Check if action is "left"
    elif action == actions[2]:
        next_state = [i, max(j - 1, 0)]

    # Check if action is "right"
    elif action == actions[3]:
        next_state = [i, min(j + 1, world["width"] - 1)]

    # Check if action is "down"
    elif action == actions[1]:
        next_state = [min(i + 1, world["height"] - 1), j]

    else:
        assert False

    # Reward for each step
    reward = -1

    # Check agent's transition into the region marked “The Cliff”
    if (action == actions[1] and i == 2 and 1 <= j <= 10) or (action == actions[3] and state == start):
        reward = -100
        next_state = start

    return next_state, reward

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

def sarsa(action_value_estimates, expected=False, step_size=step_size):
    # region Summary
    """
    An episode with SARSA
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param expected: If True, use Expected SARSA algorithm
    :param step_size: Step-size parameter for updating estimates
    :return: Total rewards within this episode
    """
    # endregion Body

    # region Body

    # Initialize state at the start
    state = start

    # Choose an action based on 𝜀-greedy algorithm
    action = choose_action(action_value_estimates, state)

    # Set the total rewards to 0
    total_rewards = 0.0

    # Keep going until getting to the goal state
    while state != goal:

        # get the next state and reward
        next_state, reward = step(state, action)

        # choose the next action
        next_action = choose_action(action_value_estimates, next_state)

        # add the obtained reward to total rewards
        total_rewards += reward

        # use SARSA
        if not expected:
            # form the target of SARSA update (Equation (6.7))
            target = action_value_estimates[next_state[0], next_state[1], next_action]

        # use Expected SARSA
        else:
            # form the target of Expected SARSA update (Equation (6.9))
            target = 0.0

            # get the next action-value estimate
            next_action_value_estimate = action_value_estimates[next_state[0], next_state[1], :]

            # get the greedy actions
            greedy_actions = np.argwhere(next_action_value_estimate == np.max(next_action_value_estimate))

            # calculate the expected value of next state
            for action_ in actions:
                if action_ in greedy_actions:
                    # greedy action selection
                    target += (((1.0 - exploration_probability) / len(greedy_actions)
                           + exploration_probability / len(actions))
                           * action_value_estimates[next_state[0],next_state[1], action_])
                else:
                    # non-greedy action selection
                    target += (exploration_probability / len(actions)
                               * action_value_estimates[next_state[0],next_state[1], action_])

        # multiply target by discount
        target *= discount

        # update action-value estimate
        action_value_estimates[state[0], state[1], action] += (step_size * (reward + target - action_value_estimates[state[0], state[1], action]))

        # move to the next state
        state = next_state

        # move to the next action
        action = next_action

    return total_rewards

    # endregion Body

def q_learning(action_value_estimates, step_size=step_size):
    # region Summary
    """
    An episode with Q-Learning
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param step_size: Step-size parameter for updating estimates
    :return: Total rewards within this episode
    """
    # endregion Summary

    # region Body

    # Initialize state at the start
    state = start

    # Set the total rewards to 0
    total_rewards = 0.0

    # Keep going until getting to the goal state
    while state != goal:

        # choose an action based on 𝜀-greedy algorithm
        action = choose_action(action_value_estimates, state)

        # get the next state and reward
        next_state, reward = step(state, action)

        # add the obtained reward to total rewards
        total_rewards += reward

        # Q-Learning update (Equation (6.8))
        action_value_estimates[state[0], state[1], action] += step_size * (reward
                            + discount * np.max(action_value_estimates[next_state[0], next_state[1], :])
                            - action_value_estimates[state[0], state[1], action])

        # move to the next state
        state = next_state

    return total_rewards

    # endregion Body

def print_optimal_policy(action_value_estimates):
    # region Summary
    """
    Print the optimal policy
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    """
    # endregion Summary

    # region Body

    # Create an empty list for optimal policy
    optimal_policy = []

    for i in range(0, world["height"]):
        # append an empty list to the optimal policy list
        optimal_policy.append([])

        for j in range(0, world["width"]):
            # check if the goal state has been reached
            if [i, j] == goal:
                optimal_policy[-1].append('G')
                continue

            # get the best action
            best_action = np.argmax(action_value_estimates[i, j, :])

            # check if best action is "up"
            if best_action == actions[0]:
                optimal_policy[-1].append('↑')

            # check if best action is "down"
            elif best_action == actions[1]:
                optimal_policy[-1].append('↓')

            # check if best action is "left"
            elif best_action == actions[2]:
                optimal_policy[-1].append('←')

            # check if best action is "right"
            elif best_action == actions[3]:
                optimal_policy[-1].append('→')

    for row in optimal_policy:
        print(row)

    # endregion Body

# endregion Functions
