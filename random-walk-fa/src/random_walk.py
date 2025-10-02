import numpy as np

# region Hyper-parameters

# Number of non-terminal states
states_number = 1000

# Non-terminal states
states = np.arange(1, states_number + 1)

# Start from a central state
start = 500

# Terminal states
terminal_states = [0, states_number + 1]

# Possible actions (left = -1, right = 1)
actions = [-1, 1]

# Maximum stride for an action
step_range = 100

# endregion Hyper-parameters

# region Helper Functions

def compute_true_value():
    # region Summary
    """
    Compute the true value of states.
    :return: States' true value
    """
    # endregion Summary

    # region Body

    print("Started computing the true values of states. Please wait.")

    # True state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # DP to find the true state values, based on the promising guess above
    # Assume all rewards are 0, termination on the left produces a reward of -1, and termination on the right produces a reward of +1
    while True:
        # Save the old value
        old_value = np.copy(true_value)

        # For every state
        for state in states:
            # set the state's true value to 0
            true_value[state] = 0

            # for every action
            for action in actions:
                # for every step
                for step in range(1, step_range + 1):
                    # multiply step by action
                    step *= action

                    # get the next state
                    next_state = state + step

                    # correct the next state
                    next_state = max(min(next_state, states_number + 1), 0)

                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * step_range) * true_value[next_state]

        # calculate the error
        error = np.sum(np.abs(old_value - true_value))

        # check convergence
        if error < 1e-2:
            break

    # Set the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    print("Finished computing the true values of states. You should see the progress bar below.")

    return true_value

    # endregion Body

def step(state, action):
    # region Summary
    """
    Take an action at state.
    :param state: State
    :param action: Action
    :return: Next state and reward for this transition
    """
    # endregion Summary

    # region Body

    # Choose a random step
    step = np.random.randint(1, step_range + 1)

    # Multiply step by action
    step *= action

    # Add step to state
    state += step

    # Correct the state
    state = max(min(state, states_number + 1), 0)

    # Termination on the left produces a reward of -1
    if state == 0:
        reward = -1

    # Termination on the right produces a reward of +1
    elif state == states_number + 1:
        reward = 1

    # All other transitions have a reward of 0
    else:
        reward = 0

    return state, reward

    # endregion Body

def get_action():
    # region Summary
    """
    Get an action, following random policy
    :return: Action
    """
    # endregion Summary

    # region Body

    if np.random.binomial(n=1, p=0.5) == 1:
        return 1

    return -1

    # endregion Body

# endregion Helper Functions

# region Value Function Classes

# region Aggregation

class ValueFunction:
    # region Summary
    """
    A wrapper class for aggregation VF
    """
    # endregion Summary

    # region Constructor

    def __init__(self, num_of_groups):
        # region Summary
        """
        Constructor of ValueFunction class
        :param num_of_groups: Number of aggregations
        """
        # endregion Summary

        # region Body

        self.num_of_groups = num_of_groups

        self.group_size = states_number // num_of_groups

        # Thetas
        self.params = np.zeros(num_of_groups)

        # endregion Body

    # endregion Constructor

    # region Functions

    def value(self, state):
        # region Summary
        """
        Get the value of state.
        :param state: State.
        :return: State's value
        """
        # endregion Summary

        # region Body

        # Check if state is terminal
        if state in terminal_states:
            return 0

        # Calculate group index
        group_index = (state - 1) // self.group_size

        return self.params[group_index]

        # endregion Body

    def update(self, update_size, state):
        # region Summary
        """
        Update parameters
        :param update_size: Size of update = step size * (target - old estimation)
        :param state: State of current sample
        """
        # endregion Summary

        # region Body

        # Calculate group index
        group_index = (state - 1) // self.group_size

        self.params[group_index] += update_size

        # endregion Body

    # endregion Functions

# endregion Aggregation

# region Polynomial and Fourier bases

basis_types = dict(polynomial = 0, fourier = 1)

class BasesValueFunction:
    # region Summary
    """
    A wrapper class for polynomial and Fourier-based VF
    """
    # endregion Summary

    # region Constructor

    def __init__(self, order, basis_type):
        # region Summary
        """
        Constructor if BasesValueFunction class
        :param order: Number of bases, each function also has 1 more constant parameter (called bias in ML)
        :param basis_type: Polynomial bases or Fourier bases
        """
        # endregion Summary

        # region Body

        self.order = order

        # Vector of feature weights (denoted as w)
        self.weights = np.zeros(order + 1)

        # Set up bases function
        self.bases = []

        # Polynomial Basis
        if basis_type == basis_types["polynomial"]:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))

        # Fourier Basis
        elif basis_type == basis_types["fourier"]:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

        # endregion Body

    # endregion Constructor

    # region Functions

    def value(self, state):
        # region Summary
        """
        Get the value of state
        :param state: Current state.
        :return: State's value
        """
        # endregion Summary

        # region Body

        # Map the state space into [0; 1]
        state /= float(states_number)

        # Get the feature vector (denoted as x(s))
        feature = np.asarray([func(state) for func in self.bases])

        # Linear methods approximate the state-value function by the inner product between w and x(s) (denoted as v^(s, w))
        state_value = np.dot(self.weights, feature)

        return state_value

        # endregion Body

    def update(self, delta, state):
        # region Summary
        """
        Update the weights
        :param delta: Update size
        :param state: Current state
        """
        # endregion Summary

        # region Body

        # Map the state space into [0; 1]
        state /= float(states_number)

        # Get derivative value
        derivative_value = np.asarray([func(state) for func in self.bases])

        # Update weights
        self.weights += delta * derivative_value

        # endregion Body

    # endregion Functions

# endregion Polynomial and Fourier bases

# region Tile Coding

class TilingsValueFunction:
    # region Summary
    """
    A wrapper class for tile coding VF
    """
    # endregion Summary

    # region Constructor

    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        # region Summary
        """
        Constructor of TilingsValueFunction class
        :param numOfTilings: Number of tilings
        :param tileWidth: Each tiling has several tiles, this parameter specifies the width of each tile
        :param tilingOffset: Specifies how tilings are put together
        """
        # endregion Summary

        # region Body

        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles, we need 1 more tile for each tiling
        self.tilingSize = states_number // tileWidth + 1

        # Weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have 1 more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

        # endregion Body

    # endregion Constructor

    # region Functions

    def value(self, state):
        # region Summary
        """
        Get the value of state
        :param state: Current state.
        :return: State's value
        """
        # endregion Summary

        # region Body

        # Initialize the state-value
        state_value = 0.0

        # Go through all the tilings
        for tiling_index in range(len(self.tilings)):
            # find the active tile in current tiling
            tile_index = (state - self.tilings[tiling_index]) // self.tileWidth

            # compute state-value
            state_value += self.params[tiling_index, tile_index]

        return state_value

        # endregion Body

    def update(self, delta, state):
        # region Summary
        """
        Update parameters
        :param delta: step-size * (target - old estimation)
        :param state: State of current sample
        """
        # endregion Summary

        # region Body

        # Each state is covered by same number of tilings => the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings

        # Go through all the tilings
        for tiling_index in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tile_index = (state - self.tilings[tiling_index]) // self.tileWidth

            # update params
            self.params[tiling_index, tile_index] += delta

        # endregion Body

    # endregion Functions

# endregion Tile Coding

# endregion Value Function Classes

# region Gradient Algorithms

def gradient_monte_carlo(value_function, step_size, states_distribution=None):
    # region Summary
    """
    Gradient MC Algorithm
    :param value_function: An instance of ValueFunction class
    :param step_size: Step-size parameter (denoted as 𝛼)
    :param states_distribution: States distribution (denoted as µ)
    """
    # endregion Summary

    # region Body

    # Start at the start state
    state = start

    # Create states trajectory
    trajectory = [state]

    # Assume discount factor: 𝛾 = 1 => return is just the same as the latest reward
    reward = 0.0

    #while episode is not over
    while state not in terminal_states:
        # Get action
        action = get_action()

        # Get the next state and reward
        next_state, reward = step(state, action)

        # Append the next state to the states trajectory
        trajectory.append(next_state)

        # Move to the next state
        state = next_state

    # Gradient update for every state in states trajectory
    for state in trajectory[:-1]:
        # calculate the update size
        update_size = step_size * (reward - value_function.value(state))

        # update VF parameters
        value_function.update(update_size, state)

        if states_distribution is not None:
            states_distribution[state] += 1



    # endregion Body

def semi_gradient_temporal_difference(value_function, steps_number, step_size):
    # region Summary
    """
    Semi-gradient n-step TD Algorithm
    :param value_function: An instance of ValueFunction class
    :param steps_number: Number of steps
    :param step_size: Step-size parameter
    """
    # endregion Summary

    # region Body

    # Initial starting state
    state = start

    # List to store states for an episode
    states = [state]

    # List to store rewards for an episode
    rewards = [0]

    # Track the time
    time_step = 0

    # The length of this episode
    episode_length = float('inf')

    while True:
        # go to next time step
        time_step += 1

        # if episode is not over
        if time_step < episode_length:
            # choose an action randomly
            action = get_action()

            # get the next state and reward
            next_state, reward = step(state, action)

            # append the new state to the list
            states.append(next_state)

            # append the reward to the list
            rewards.append(reward)

            # if the next state is terminal state
            if next_state in terminal_states:
                # end the episode
                episode_length = time_step

        # get the time of the state to update
        update_time = time_step - steps_number


        if update_time >= 0:
            # prepare to calculate returns
            returns = 0.0

            # calculate corresponding rewards
            for t in range(update_time + 1, min(episode_length, update_time+steps_number) + 1):
                # add reward to the return
                returns += rewards[t]

            # if episode is not over
            if update_time + steps_number <= episode_length:

                # add approximate state value to the return
                returns += value_function.value(states[update_time + steps_number])

            # get the state to be updated
            state_to_update = states[update_time]

            # if the state to be updated is not a terminal state
            if not state_to_update in terminal_states:
                # calculate the update size
                update_size = step_size * (returns - value_function.value(state_to_update))

                # update the VF
                value_function.update(update_size, state_to_update)

        # check if episode ended
        if update_time == episode_length - 1:
            break

        # move to the next state
        state = next_state

    # endregion Body

# endregion Gradient Algorithms
