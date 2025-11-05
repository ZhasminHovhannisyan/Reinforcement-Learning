import numpy as np
from tqdm import tqdm
from src.tile_coding import IHT, tiles

# region Hyper-parameters

# Possible priorities
priorities = np.arange(0, 4)

# Reward for each priority
rewards = np.power(2, np.arange(0, 4))

# Possible actions
actions = dict(reject = 0, accept = 1)

# Total number of servers
number_of_servers = 10

# At each time step, a busy server will be free with probability 0.06 (denoted as p)
probability_free = 0.06

# Step-size parameter for learning state-action value (denoted as 𝛼)
step_size_state_action_value = 0.01

# Step-size parameter for learning average reward (denoted as 𝛽)
step_size_average_reward = 0.01

# Exploration probability (denoted as 𝜀)
exploration_probability = 0.1

# endregion Hyper-parameters

# region Helpers

def get_action(free_servers, priority, value_function):
    # region Summary
    """
    Get action at given state (number of free servers and customer priority) based on ε-greedy policy and given VF
    :param free_servers: Number of free servers
    :param priority: Customer priority
    :param value_function: VF
    :return: Action
    """
    # endregion Summary

    # region Body

    # If no free server,
    if free_servers == 0:
        # can't accept <=> reject
        return actions["reject"]

    # ε-greedy action selection: every once in a while, with small probability ε, select randomly from among all the actions with equal probability, independently of the action-value estimates.
    if np.random.binomial(n=1, p=exploration_probability) == 1:
        return np.random.choice(list(actions.values()))

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    values = [value_function.state_action_value(free_servers, priority, action) for action in list(actions.values())]
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

    # endregion Body

def take_action(free_servers, priority, action):
    # region Summary
    """
    Take an action at state (number of free servers and customer priority)
    :param free_servers: Number of free servers
    :param priority: Customer priority
    :param action: Current action
    :return: New number of free servers, new priority, reward
    """
    # endregion Summary

    # region Body

    # If there are free servers and customer is accepted
    if free_servers > 0 and action == actions["accept"]:
        # decrement number of free servers
        free_servers -= 1

    # Calculate reward based on customer priority
    reward = rewards[priority] * action

    # Some busy servers may become free
    busy_servers = number_of_servers - free_servers
    free_servers += np.random.binomial(n=busy_servers, p=probability_free)

    return free_servers, np.random.choice(priorities), reward

    # endregion Body


class ValueFunction:
    # region Summary
    """
    A wrapper class for differential semi-gradient SARSA state-action VF
    NOTE: in this example, a tiling software is used instead of implementing custom standard tiling.
    One important thing is that tiling is only a map from (state, action) to a series of indices.
    It doesn't matter whether the indices have meaning, only if this map satisfy some property.
    View the following webpage for more information: http://www.incompleteideas.net/tiles.html
    """
    # endregion Summary

    # region Constructor

    def __init__(self, num_of_tilings, ss_state_action_value=step_size_state_action_value, ss_average_reward=step_size_average_reward):
        # region Summary
        """
        Constructor of ValueFunction class
        :param num_of_tilings: Number of tilings
        :param ss_state_action_value: Step-size parameter for learning state-action value (denoted as 𝛼)
        :param ss_average_reward: Step-size parameter for learning average reward (denoted as 𝛽)
        """
        # endregion Summary

        # region Body

        self.num_of_tilings = num_of_tilings

        # Divide step-size for learning state-action value equally to each tiling
        self.ss_state_action_value = ss_state_action_value / self.num_of_tilings

        self.ss_average_reward = ss_average_reward

        # The maximum number of indices
        self.max_size = 2048

        # Hash table
        self.hash_table = IHT(self.max_size)

        # Weight for each tile
        self.weights = np.zeros(self.max_size)

        # State features (server and priority) need scaling to satisfy the tile software
        self.server_scale = self.num_of_tilings / float(number_of_servers)
        self.priority_scale = self.num_of_tilings / float(len(priorities) - 1)

        # Initialize average reward with 0
        self.average_reward = 0.0

        # endregion Body

    # endregion Constructor

    # region Functions

    def get_active_tiles(self, free_servers, priority, action):
        # region Summary
        """
        Get indices of active tiles for given state and action
        :param free_servers: Number of free servers
        :param priority: Customer priority
        :param action: Current action
        :return: Active tiles
        """
        # endregion Summary

        # region Body

        active_tiles = tiles(iht_or_size=self.hash_table,
                             num_tilings=self.num_of_tilings,
                             floats=[self.server_scale * free_servers, self.priority_scale * priority],
                             ints=[action])

        return active_tiles

        # endregion Body

    def state_action_value(self, free_servers, priority, action):
        # region Summary
        """
        Estimate the value of given state and action without subtracting average
        :param free_servers: Number of free servers
        :param priority: Customer priority
        :param action: Given action
        :return: State-action value estimate without subtracting average
        """
        # endregion Summary

        # region Body

        # Get indices of active tiles for given state and action
        active_tiles = self.get_active_tiles(free_servers, priority, action)

        # Calculate state-action value estimate
        state_action_value_estimate = np.sum(self.weights[active_tiles])

        return state_action_value_estimate

        # endregion Body

    def state_value(self, free_servers, priority):
        # region Summary
        """
        Estimate the value of given state without subtracting average
        :param free_servers: Number of free servers
        :param priority: Customer priority
        :return: State-value estimate without subtracting average
        """
        # endregion Summary

        # region Body

        # Estimate state-action values
        state_action_values = [self.state_action_value(free_servers, priority, action) for action in list(actions.values())]

        # If no free server,
        if free_servers == 0:
            # can't accept <=> reject
            return state_action_values[actions["reject"]]

        # Calculate state-value estimate
        state_value = np.max(state_action_values)

        return state_value

        # endregion Body

    def learn(self, free_servers, priority, action, new_free_servers, new_priority, new_action, reward):
        # region Summary
        """
        Learn with given sequence
        :param free_servers: Number of free servers
        :param priority: Customer priority
        :param action: Given action
        :param new_free_servers: New number of free servers
        :param new_priority: New customer priority
        :param new_action: New action
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Get indices of active tiles for given state and action
        active_tiles = self.get_active_tiles(free_servers, priority, action)

        # Calculate value estimate
        estimation = np.sum(self.weights[active_tiles])

        # Calculate update size
        update_size = reward - self.average_reward + self.state_action_value(new_free_servers, new_priority, new_action) - estimation

        # Update average reward
        self.average_reward += self.ss_average_reward * update_size

        # Calculate new update size
        update_size *= self.ss_state_action_value

        # For every active tile
        for active_tile in active_tiles:
            # update active tile's weight
            self.weights[active_tile] += update_size

        # endregion Body

    # endregion Functions

# endregion Helpers

# region Functions

def differential_semi_gradient_sarsa(value_function, max_steps):
    # region Summary
    """
    Differential semi-gradient SARSA
    :param value_function: State-value function to learn
    :param max_steps: Step limit in the continuing task
    """
    # endregion Summary

    # region Body

    # Get the currently free serves
    current_free_servers = number_of_servers

    # Get current customer priority
    current_priority = np.random.choice(priorities)

    # Get initial action
    current_action = get_action(current_free_servers, current_priority, value_function)

    # Track the hit for each number of free servers
    freq = np.zeros(number_of_servers + 1)

    # For every time step
    for _ in tqdm(range(max_steps)):

        # increment frequency of currently free servers
        freq[current_free_servers] += 1

        # take an action at current state
        new_free_servers, new_priority, reward = take_action(current_free_servers, current_priority, current_action)

        # get a new action
        new_action = get_action(new_free_servers, new_priority, value_function)

        # learn VF with given sequence
        value_function.learn(current_free_servers, current_priority, current_action,
                             new_free_servers, new_priority, new_action, reward)

        # move to the next state
        current_free_servers = new_free_servers
        current_priority = new_priority

        # move to the next action
        current_action = new_action

    print(f"Frequency of number of free servers: {freq/max_steps}")



    # endregion Body

# endregion Functions
