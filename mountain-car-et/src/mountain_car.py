import numpy as np

from src.tile_coding import IHT, tiles

# region Hyper-parameters

# All possible actions (order is important)
actions = dict(reverse=-1, zero=0, forward=1)

# Bounds for position
POSITION = dict(min=-1.2, max=0.5)

# Bounds for velocity
VELOCITY = dict(min=-0.07, max=0.07)

# Discount (denoted as 𝛾) is always 1.0 in these experiments
discount = 1.0

# Use optimistic initial value, so it's OK to set ε = 0
exploration_probability = 0

# Maximum steps per episode
step_limit = 5000

# endregion Hyper-parameters

# region Helpers

def step(position, velocity, action):
    # region Summary
    """
    Take an action at state (position and velocity)
    :param position: Current position
    :param velocity: Current velocity
    :param action: Current action
    :return: New position, new velocity, reward (always -1)
    """
    # endregion Summary

    # region Body

    # Calculate new velocity
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY["min"], new_velocity), VELOCITY["max"])

    # Calculate new position
    new_position = position + new_velocity
    new_position = min(max(POSITION["min"], new_position), POSITION["max"])

    # The reward in this problem is -1 on all time steps until the car moves past its goal position
    reward = -1.0

    # When position reached the left bound,
    if new_position == POSITION["min"]:
        # velocity was reset to 0
        new_velocity = 0.0

    return new_position, new_velocity, reward

    # endregion Body

def get_action(position, velocity, value_function):
    # region Summary
    """
    Get action at given state (position and velocity) based on ε-greedy policy and given VF
    :param position: Current position
    :param velocity: Current velocity
    :param value_function: VF
    :return: Action
    """
    # endregion Summary

    # region Body

    # ε-greedy action selection: every once in a while, with small probability ε, select randomly from among all the actions with equal probability, independently of the action-value estimates.
    if np.random.binomial(n=1, p=exploration_probability) == 1:
        return np.random.choice(list(actions.values()))

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    values = []
    for action in list(actions.values()):
        values.append(value_function.value(position, velocity, action))
    return np.argmax(values) - 1

    # endregion Body

# region Eligibility Traces

def accumulating_trace(trace, active_tiles, trace_decay):
    # region Summary
    """
    Update rule for accumulating trace (Equation (12.5))
    :param trace: Old eligibility trace (denoted as 𝒛)
    :param active_tiles: Current active tile indices
    :param trace_decay: Trace-decay parameter (denoted as 𝜆)
    :return: New eligibility trace for convenience
    """
    # endregion Summary

    # region Body

    # Update the ET vector
    trace *= discount * trace_decay
    trace[active_tiles] += 1

    return trace

    # endregion Body

def dutch_trace(trace, active_tiles, trace_decay, step_size):
    # region Summary
    """
    Update rule for Dutch trace (Equation (12.11))
    :param trace: Old eligibility trace (denoted as 𝒛)
    :param active_tiles: Current active tile indices
    :param trace_decay: Trace-decay parameter (denoted as 𝜆)
    :param step_size: Step-size parameter (denoted as 𝛼) for all tiles
    :return: New eligibility trace for convenience
    """
    # endregion Summary

    # region Body

    # Calculate the coefficient of 2nd additive in the Equation (12.11)
    coefficient = 1 - step_size * discount * trace_decay * np.sum(trace[active_tiles])

    # Update the ET vector
    trace *= discount * trace_decay
    trace[active_tiles] += coefficient

    return trace

    # endregion Body

def replacing_trace(trace, active_tiles, trace_decay):
    # region Summary
    """
    Update rule for replacing trace (Equation (12.12))
    :param trace: Old eligibility trace (denoted as 𝒛)
    :param active_tiles: Current active tile indices
    :param trace_decay: Trace-decay parameter (denoted as 𝜆)
    :return: New eligibility trace for convenience
    """
    # endregion Summary

    # region Body

    # The replacing trace is defined on a component-by-component basis depending on whether the component of the feature vector was:
    active = np.in1d(np.arange(len(trace)), active_tiles)

    # a. active (= 1),
    trace[active] = 1

    # b. inactive (= 0)
    trace[~active] *= discount * trace_decay

    return trace

    # endregion Body

def replacing_trace_with_clearing(trace, active_tiles, trace_decay, clearing_tiles):
    # region Summary
    """
    Update rule for replacing trace with clearing ("clearing" means set all tiles corresponding to non-selected actions to 0)
    :param trace: Old eligibility trace (denoted as 𝒛)
    :param active_tiles: Current active tile indices
    :param trace_decay: Trace-decay parameter (denoted as 𝜆)
    :param clearing_tiles: Tiles to be cleared
    :return: New eligibility trace for convenience
    """
    # endregion Summary

    # region Body

    # The replacing trace is defined on a component-by-component basis depending on whether the component of the feature vector was:
    active = np.in1d(np.arange(len(trace)), active_tiles)

    # a. inactive (= 0),
    trace[~active] *= discount * trace_decay

    # b. to be cleared,
    trace[clearing_tiles] = 0

    # c. active (= 1)
    trace[active] = 1

    return trace

    # endregion Body

# endregion Eligibility Traces


class SARSA:
    # region Summary
    """
    A wrapper class for SARSA(𝜆)
    NOTE: in this example, a tiling software is used instead of implementing custom standard tiling.
    One important thing is that tiling is only a map from (state, action) to a series of indices.
    It doesn't matter whether the indices have meaning, only if this map satisfies some property.
    View the following web page for more information: http://www.incompleteideas.net/tiles.html
    """
    # endregion Summary

    # region Constructor

    def __init__(self, step_size, trace_decay, trace_update=accumulating_trace, num_of_tilings=8, max_size=2048):
        # region Summary
        """
        Constructor of SARSA class
        :param step_size: Step-size parameter (denoted as 𝛼)
        :param trace_decay: Trace-decay parameter (denoted as 𝜆)
        :param trace_update: Eligibility trace type (accumulating, Dutch, replacing)
        :param num_of_tilings: Number of tilings
        :param max_size: The maximum number of indices
        """
        # endregion Summary

        # region Body

        # Divide step-size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.trace_decay = trace_decay
        self.trace_update = trace_update
        self.num_of_tilings = num_of_tilings
        self.max_size = max_size

        # Hash table
        self.hash_table = IHT(max_size)

        # Weight for each tile
        self.weights = np.zeros(max_size)

        # Trace for each tile
        self.trace = np.zeros(max_size)

        # State features (position and velocity) need scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION["max"] - POSITION["min"])
        self.velocity_scale = self.num_of_tilings / (VELOCITY["max"] - VELOCITY["min"])

        # endregion Body

    # endregion Constructor

    # region Functions

    def get_active_tiles(self, position, velocity, action):
        # region Summary
        """
        Get indices of active tiles for given state and action
        :param position: Current position
        :param velocity: Current velocity
        :param action: Given action
        :return: Active tiles
        """
        # endregion Summary

        # region Body

        # Probably, position_scale * (position - position_min) would be a good normalization.
        # However, position_scale * position_min is a constant, so it's OK to ignore it.
        active_tiles = tiles(iht_or_size=self.hash_table,
                             num_tilings=self.num_of_tilings,
                             floats=[self.position_scale * position, self.velocity_scale * velocity],
                             ints=[action])

        return active_tiles

        # endregion Body

    def value(self, position, velocity, action):
        # region Summary
        """
        Estimate the value of given state and action
        :param position: Current position
        :param velocity: Current velocity
        :param action: Given action
        :return: Value estimate
        """
        # endregion Summary

        # region Body

        # When position reached the right bound,
        if position == POSITION["max"]:
            # the goal was reached and the episode was terminated
            value_estimate = 0.0

        else:
            # Get indices of active tiles for given state and action
            active_tiles = self.get_active_tiles(position, velocity, action)

            # Calculate value estimate
            value_estimate = np.sum(self.weights[active_tiles])

        return value_estimate

        # endregion Body

    def learn(self, position, velocity, action, target):
        # region Summary
        """
        Learn with given state, action and target
        :param position: Current position
        :param velocity: Current velocity
        :param action: Given action
        :param target: Given target
        """
        # endregion Summary

        # region Body

        # Get indices of active tiles for given state and action
        active_tiles = self.get_active_tiles(position, velocity, action)

        # Calculate value estimate
        value_estimation = np.sum(self.weights[active_tiles])

        # Calculate TD error (denoted as 𝛿)
        TD_error = target - value_estimation

        # If ET is accumulating or replacing
        if self.trace_update == accumulating_trace or self.trace_update == replacing_trace:
            # update ET according to the corresponding rule
            self.trace_update(self.trace, active_tiles, self.trace_decay)

        # If ET is Dutch
        elif self.trace_update == dutch_trace:
            # update ET according to the corresponding rule
            self.trace_update(self.trace, active_tiles, self.trace_decay, self.step_size)

        # If ET is replacing with tiles clearing
        elif self.trace_update == replacing_trace_with_clearing:
            # create an empty list for clearing tiles
            clearing_tiles = []

            # for every action
            for act in list(actions.values()):
                # if action wasn't selected
                if act != action:
                    # extend the list of clearing tiles with active tiles
                    clearing_tiles.extend(self.get_active_tiles(position, velocity, act))

            # update ET according to the corresponding rule
            self.trace_update(self.trace, active_tiles, self.trace_decay, clearing_tiles)

        else:
            raise Exception("Unexpected Trace Type")

        # Update weights (equation on page 303)
        self.weights += self.step_size * TD_error * self.trace

        # endregion Body

    def cost_to_go(self, position, velocity):
        # region Summary
        """
        Get number of steps to reach the goal under current state-value function
        :param position: Current position
        :param velocity: Current velocity
        :return: Number of steps to reach the goal
        """
        # endregion Summary

        # region Body

        # Create an empty list for costs
        costs = []

        # For every action
        for action in list(actions.values()):
            # append the value estimate of given state and action to the list of costs
            costs.append(self.value(position, velocity, action))

        return -np.max(costs)

        # endregion Body

    # endregion Functions

# endregion Helpers

# region Functions

def play(evaluator):
    # region Summary
    """
    Play for 1 episode based on given method evaluator
    :param evaluator: Given method
    :return: Total steps in this episode
    """
    # endregion Summary

    # region Body

    # Start at a random position around the bottom of the valley
    current_position = np.random.uniform(-0.6, -0.4)

    # Start with 0 initial velocity
    current_velocity = 0.0

    # Get initial action
    current_action = get_action(current_position, current_velocity, evaluator)

    # Track the steps
    steps = 0

    while True:
        # Take current action and move to the next state
        next_position, next_velocity, reward = step(current_position, current_velocity, current_action)

        # Choose next action
        next_action = get_action(next_position, next_velocity, evaluator)

        # Increment steps
        steps += 1

        # Calculate target
        target = reward + discount *evaluator.value(next_position, next_velocity, next_action)

        # Learn with given state, action and target
        evaluator.learn(current_position, current_velocity, current_action, target)

        # Move to the next state
        current_position = next_position
        current_velocity = next_velocity

        # Move to the next action
        current_action = next_action

        # When position reached the right bound,
        if next_position == POSITION["max"]:
            # the goal was reached and the episode was terminated
            break

        # If step limit exceeded,
        if steps >= step_limit:
            # the episode was terminated
                print(" - Step Limit Exceeded!")
                break

    return steps
    # endregion Body

# endregion Functions
