import numpy as np

from src.tile_coding import IHT, tiles

# region Hyper-parameters

# All possible actions (order is important)
all_actions = dict(reverse=-1, zero=0, forward=1)

# Bounds for position
POSITION = dict(min=-1.2, max=0.5)

# Bounds for velocity
VELOCITY = dict(min=-0.07, max=0.07)

# Use optimistic initial value, so it's OK to set ε = 0
exploration_probability = 0

# endregion Hyper-parameters

# region Helpers

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
        return np.random.choice(list(all_actions.values()))

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    values = []
    for action in list(all_actions.values()):
        values.append(value_function.value(position, velocity, action))
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)]) - 1

    # endregion Body

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


class ValueFunction:
    # region Summary
    """
    A wrapper class for state-action VF
    NOTE: in this example, a tiling software is used instead of implementing custom standard tiling.
    One important thing is that tiling is only a map from (state, action) to a series of indices.
    It doesn't matter whether the indices have meaning, only if this map satisfies some property.
    View the following web page for more information: http://www.incompleteideas.net/tiles.html
    """
    # endregion Summary

    # region Constructor

    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        # region Summary
        """
        Constructor of ValueFunction class
        :param step_size: Step-size parameter
        :param num_of_tilings: Number of tilings
        :param max_size: The maximum number of indices
        """
        # endregion Summary

        # region Body

        # Divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.num_of_tilings = num_of_tilings
        self.max_size = max_size

        # Hash table
        self.hash_table = IHT(max_size)

        # Weight for each tile
        self.weights = np.zeros(max_size)

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

        # Calculate update size
        update_size = self.step_size * (target - value_estimation)

        # For every active tile
        for active_tile in active_tiles:
            # update active tile's weight
            self.weights[active_tile] += update_size

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
        for action in list(all_actions.values()):
            # append the value estimate of given state and action to the list of costs
            costs.append(self.value(position, velocity, action))

        return -np.max(costs)

        # endregion Body

    # endregion Functions

# endregion Helpers

# region Functions

def semi_gradient_n_step_sarsa(value_function, number_of_steps=1):
    # region Summary
    """
    Semi-gradient n-step SARSA
    :param value_function: State-value function to learn
    :param number_of_steps: Number of steps
    :return: Time step
    """
    # endregion Summary

    # region Body

    # Start at a random position around the bottom of the valley
    current_position = np.random.uniform(-0.6, -0.4)


    # Start with 0 initial velocity
    current_velocity = 0.0


    # Get initial action
    current_action = get_action(current_position, current_velocity, value_function)


    # Track previous positions
    positions = [current_position]


    # Track previous velocities
    velocities = [current_velocity]


    # Track previous actions
    actions = [current_action]


    # Track previous rewards
    rewards = [0.0]


    # Track the time step
    time_step = 0


    # Define the length of this episode (denoted as T)
    episode_length = float('inf')


    while True:

        # move to next time step
        time_step += 1


        # if episode is not over
        if time_step < episode_length:


            # take current action and move to the new state
            new_position, new_velocity, reward = step(current_position, current_velocity, current_action)

            new_action = get_action(new_position, new_velocity, value_function)
            # choose new action
            positions.append(new_position)
            velocities.append(new_velocity)
            actions.append(new_action)


            # track new state
            rewards.append(reward)


            # track new action


            # track reward


            # when position reached the right bound,
            if new_position == POSITION["max"]:

                # the goal was reached and the episode was terminated
                episode_length = time_step


        # get the time of the state to update
        update_time = time_step - number_of_steps



        if update_time >= 0:

            # initialize returns with 0
            returns = 0.0


            # calculate corresponding returns
            for t in range(update_time + 1, min(episode_length, update_time + number_of_steps) + 1):
                returns += rewards[t]


            if update_time + number_of_steps <= episode_length:



                # add estimated state-action value to the return
                returns += value_function.value(positions[update_time + number_of_steps], velocities[update_time + number_of_steps], actions[update_time + number_of_steps] )


            # update the state-value function
            if positions[update_time] != POSITION['max']:
                value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)

        if update_time == episode_length - 1:
            break


        # move to the next state
        current_position = new_position
        current_velocity = new_velocity


        # move to the next action
        current_action = new_action
    return time_step




    # endregion Body

def print_cost(value_function, episode, ax):
    # region Summary
    """
    Print learned cost to go
    :param value_function: Value Function
    :param episode: Episode
    :param ax: Axis
    """
    # endregion Summary

    # region Body

    grid_size = 40

    positions = np.linspace(POSITION["min"], POSITION["max"], grid_size)

    # positionStep = (POSITION_MAX - POSITION_MIN) / grid_size
    # positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    # velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / grid_size
    # velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)

    velocities = np.linspace(VELOCITY["min"], VELOCITY["max"], grid_size)

    axis_x = []
    axis_y = []
    axis_z = []

    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))

    ax.scatter(axis_x, axis_y, axis_z)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')

    ax.set_title('Episode %d' % (episode + 1))

    # endregion Body

# endregion Functions