import numpy as np
from tqdm import tqdm

# region Hyper-parameters

# Denote: i=0 is the left terminal state; i=1,2,3,4,5 represent the non-terminal states A,B,C,D,E; i=6 is the right terminal state.
# For convenience, we assume all rewards are 0, and the left terminal state has value 0, the right terminal state has value 1.
# This trick has been used in Gambler's Problem.

# Approximate (learned) state-values
approximate_values = np.zeros(7)

# Initialize the approximate values of non-terminal states to the intermediate value
approximate_values[1:6] = 0.5

# The approximate value of the right terminal state
approximate_values[6] = 1

# True state-values
true_values = np.zeros(7)

# The true value of each non-terminal state is the probability of terminating on the right if starting from that state
true_values[1:6] = np.arange(1, 6) / 6.0

# The true value of the right terminal state
true_values[6] = 1

# Actions
actions = dict(left = 0, right = 1)

# endregion Hyper-parameters

# region Functions

def monte_carlo(current_values, step_size=0.1, batch=False):
    # region Summary
    """
    Monte Carlo Method
    :param current_values: Current state-values, which will be updated if batch is False
    :param step_size: Step-size parameter (denoted as 𝛼)
    :param batch: Whether to update current state-values
    :return: Trajectory of states, reward
    """
    # endregion Summary

    # region Body

    # All episodes start in the center state C (i.e., i = 3)
    state = 3

    # Add the 1st state to the trajectory of states
    states_trajectory = [state]

    while True:
        # If the agent moves left
        if np.random.binomial(n=1, p=0.5) == actions["left"]:
            # the state is decremented
            state -= 1
        else:
            # the state is incremented
            state += 1

        # Append the new state to the trajectory of states
        states_trajectory.append(state)

        # If an episode terminates on the right terminal state, reward is +1
        if state == 6:
            reward = 1.0
            break

        # If an episode terminates on the left terminal state, reward is 0
        elif state == 0:
            reward = 0.0
            break

    if not batch:
        # For every state in trajectory of states (except the last one)
        for state_ in states_trajectory[:-1]:
            # MC update (Equation (6.1))
            current_values[state_] += step_size * (reward - current_values[state_])

    return states_trajectory, [reward] * (len(states_trajectory) - 1)

    # endregion Body

def temporal_difference(current_values, step_size=0.1, batch=False):
    # region Summary
    """
    Temporal-Difference Learning
    :param current_values: Current state-values, which will be updated if batch is False
    :param step_size: Step-size parameter (denoted as 𝛼)
    :param batch: Whether to update current state-values
    :return: Trajectory of states, rewards
    """
    # endregion Summary

    # region Body

    # All episodes start in the center state C (i.e., i = 3)
    state = 3

    # Add the 1st state to the trajectory of states
    states_trajectory = [state]

    # Create a list of rewards with a single 0 in it
    rewards = [0]

    while True:
        # Preserve the old state
        old_state = state

        # If the agent moves left
        if np.random.binomial(n=1, p=0.5) == actions["left"]:

            # the state is decremented
            state -= 1

        else:
            # the state is incremented
            state += 1

        # Append the new state to the trajectory of states
        states_trajectory.append(state)

        # Assume all rewards are 0
        reward = 0

        if not batch:
            # TD update (Equation (6.2))
            current_values[old_state] += step_size * (reward + current_values[state] - current_values[old_state])

        # Episode terminates either on the right terminal state or on the left terminal state
        if state ==6 or state == 0:
            break

        # Append the reward to the list of rewards
        rewards.append(reward)

    return states_trajectory, rewards

    # endregion Body

def batch_updating(method, episodes, step_size=0.001, threshold=1e-3):
    # region Summary
    """
    Batch updating
    :param method: "TD" or "MC"
    :param episodes: Number of episodes
    :param step_size: Step-size parameter (denoted as 𝛼)
    :param threshold: Small threshold determining accuracy of estimation (denoted as 𝜃 > 0)
    :return: Total errors
    """
    # endregion Summary

    # region Body

    # Perform 100 independent runs
    runs = 100

    # Create an array of total errors filled with 0s
    total_errors = np.zeros(runs)

    # For every run
    for _ in tqdm(range(runs)):

        # get the current state-values of non-terminal states
        current_values = np.copy(approximate_values)
        current_values[1:6] = -1

        # create an empty list for RMSEs
        rms_errors = []

        # create an empty list for trajectories of states
        states_trajectory = []

        # create an empty list for rewards
        rewards = []

        # for every episode
        for _ in range(episodes):

            # check the method, then get trajectory and reward
            if method == "TD":
                trajectory, reward = temporal_difference(current_values, batch=True)

            else:
                trajectory, reward = monte_carlo(current_values, batch=True)
            # append the trajectory to the list of trajectories of states
            states_trajectory.append(trajectory)

            # append the reward to the list of rewards
            rewards.append(reward)

            # keep feeding our algorithm with trajectories seen so far until state-value function converges
            while True:
                # create an empty array for updates filled with 0s
                updates = np.zeros(7)

                # for every (trajectory, reward) pair
                for trajectory, reward in zip(states_trajectory, rewards):
                    # for every trajectory
                    for i in range(len(trajectory) - 1):
                        # check the method and perform update
                        if method == "TD":
                            # TD update (Equation (6.2))
                            updates[trajectory[i]] += reward[i] + current_values[trajectory[i + 1]] - current_values[trajectory[i]]

                        else:
                            # MC update (Equation (6.1))
                            updates[trajectory[i]] += reward[i] - current_values[trajectory[i]]

                # complete the update
                updates *= step_size

                # check the state-value functon convergence
                if np.sum(np.abs(updates)) < threshold:
                    break

                # perform batch updating
                current_values += updates

            # calculate RMSE between the true state-values and current state-values, averaged over the 5 states
            rmse = np.sqrt(np.sum(np.power(current_values - true_values, 2)) / 5.0)
            # append the RMSE to the list of RMSEs
            rms_errors.append(rmse)

        # add RMSEs to total errors
        total_errors += np.asarray(rms_errors)

    # average total errors over runs
    total_errors /= runs

    return total_errors

    # endregion Body

# endregion Functions
