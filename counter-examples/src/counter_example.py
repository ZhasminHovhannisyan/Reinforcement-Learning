import numpy as np


# region Hyper-parameters

# All states: state 0-5 are upper states
states = np.arange(0, 7)

# State 6 is lower state
lower_state = 6

# Discount factor (denoted as 𝛾)
discount = 0.99

# Each state is represented by a feature vector 𝒙(𝑠) of length d = 8
feature_vector_size = 8

# Feature matrix of size number of states * feature vector size filled with 0s
features = np.zeros((len(states), feature_vector_size))

# For every upper state
for i in range(len(states) - 1):
    # set the i-th feature's value to 2 (𝑥_i(𝑠) = 2)
    features[i, i] = 2

    # set the last feature's value to 1 (𝑥_last(𝑠) = 1)
    features[i, feature_vector_size - 1] = 1

# For lower state, 𝑥_pre-last(𝑠) = 1 and 𝑥_last(𝑠) = 2
features[lower_state, feature_vector_size - 2] = 1
features[lower_state, feature_vector_size - 1] = 2

# All possible actions
actions = dict(dashed = 0, solid = 1)

# Reward is always 0
reward = 0

# The behavior policy selects the solid action with probability 1/7
behavior_solid_probability = 1.0 / 7

# State distribution (denoted as μ) for the behavior policy
state_distribution = np.ones(len(states)) / 7
state_distribution_matrix = np.matrix(np.diag(state_distribution))

# Projection matrix (denoted as Π) for minimizing Mean Squared Value Error (MS-VE)
projection_matrix = (np.matrix(features) * np.linalg.pinv(np.matrix(features.T) * state_distribution_matrix * np.matrix(features))
                                         * np.matrix(features.T) * state_distribution_matrix)

# Interest is 1 for every state
interest = 1

# endregion Hyper-parameters

# region Functions

# region Helpers

def step(state, action):
    # region Summary
    """
    Take action at current state.
    :param state: Current state
    :param action: Action
    :return: New state
    """
    # endregion Summary

    # region Body

    # The solid action takes the system to the 7th (lower) state
    if action == actions["solid"]:
        new_state = lower_state

    # The dashed action takes the system to 1 of the 6 upper states with equal probability
    else:
        new_state = np.random.choice(states[: lower_state])

    return new_state

    # endregion Body

# region Policies

def target_policy(state):
    # region Summary
    """
    Action selection according to the target policy
    :param state: Current state
    :return: Action
    """
    # endregion Summary

    # region Body

    # The target policy always takes the solid action => the on-policy distribution is concentrated in the 7th (lower) state
    return actions["solid"]

    # endregion Body

def behavior_policy(state):
    # region Summary
    """
    Action selection according to the behavior policy
    :param state: Current state
    :return: Action
    """
    # endregion Summary

    # region Body

    # The behavior policy selects the solid action with probability 1/7,
    if np.random.binomial(n=1, p=behavior_solid_probability) == 1:
        action = actions["solid"]

    # and the dashed action with probability 6/7
    else:
        action = actions["dashed"]

    return action

    # endregion Body

# endregion Policies

# region Error calculations

def compute_RMSVE(weights):
    # region Summary
    """
    Compute Root Mean Square Value Error (RMS-VE) for a VF parameterized by weights
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :return: RMS-VE
    """
    # endregion Summary

    # region Body

    # Calculate the error in each state, i.e., the square of the difference between the approximate VF 𝑣 ̂(𝑠,𝒘) and the true VF 𝑣_𝜋(𝑠). In this example:
    # 1. 𝑣 ̂(𝑠,𝒘) = 𝒘^𝑇 ∙ 𝒙(𝑠),
    # 2. 𝑣_𝜋(𝑠) = 0.
    difference = np.power(np.dot(features, weights), 2)

    # Weighting this difference over the state space by μ, we obtain a natural objective function, the MS-VE (Equation (9.1))
    ms_ve = np.dot(difference, state_distribution)

    # The square root of MS-VE, the RMS-VE, gives a rough measure of how much the approximate values differ from the true values
    rms_ve = np.sqrt(ms_ve)

    return rms_ve

    # endregion Body

def compute_RMSPBE(weights):
    # region Summary
    """
    Compute Root Mean Square Projected Bellman Error (RMS-PBE) for a VF parameterized by weights
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :return: RMS-PBE
    """
    # endregion Summary

    # region Body

    # Initialize the BEs with 0s
    bellman_errors = np.zeros(len(states))

    # For every state
    for state in states:
        # for every next state
        for next_state in states:
            # if next state is the lower state
            if next_state == lower_state:
                # calculate BE
                bellman_errors[state] += reward + discount * np.dot(weights, features[next_state, :]) - np.dot(weights, features[state, :])

    # Calculate projected BEs
    bellman_errors = np.dot(np.asarray(projection_matrix), bellman_errors)

    # Calculate the error in each state, i.e., the square of the difference between the BE and the true VF which is 0 in this example
    difference = np.power(bellman_errors, 2)

    # Weighting this difference over the state space by μ, will give MS-PBE
    ms_pbe = np.dot(difference, state_distribution)

    # The square root of MS-PBE is RMS-PBE
    rms_pbe = np.sqrt(ms_pbe)

    return rms_pbe

    # endregion Body

# endregion Error calculations

# endregion Helpers

# region Semi-gradient Methods

def semi_gradient_off_policy_TD(state, weights, step_size):
    # region Summary
    """
    Semi-gradient off-policy TD
    :param state: Current state
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :param step_size: Step-size parameter (denoted as 𝛼)
    :return: Next state
    """
    # endregion Summary

    # region Body

    # Select an action according to the behavior policy
    action = behavior_policy(state)

    # Get the next state
    next_state = step(state, action)

    # Calculate the importance sampling ratio (denoted as 𝜌) according to the Equation (11.1)
    if action == actions["dashed"]:
        importance_sampling_ratio = 0.0
    else:
        importance_sampling_ratio = 1.0 / behavior_solid_probability

    # Calculate TD error (denoted as 𝛿) according to the Equation (11.3) for linear case, i.e., 𝑣 ̂(𝑠,𝒘) = 𝒘^𝑇 ∙ 𝒙(𝑠)
    TD_error = reward + discount * np.dot(features[next_state, :], weights) - np.dot(features[state,:], weights)

    # Update weights according to the Equation (11.2). Derivatives happen to be the same matrix due to the linearity
    weights += step_size * importance_sampling_ratio * TD_error * features[state, :]

    return next_state


    # endregion Body

def semi_gradient_DP(weights, step_size):
    # region Summary
    """
    Semi-gradient DP
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :param step_size: Step-size parameter (denoted as 𝛼)
    """
    # endregion Summary

    # region Body

    # Initialize error
    error = 0.0

    # Go through all the states
    for state in states:
        # initialize the expected return
        expected_return = 0.0

        # for every state
        for next_state in states:
            if next_state == lower_state:
                # compute the expected return
                expected_return += reward + discount * np.dot(weights, features[next_state, :])

        # calculate Bellman error for every state
        bellman_error = expected_return - np.dot(weights, features[state, :])

        # accumulate gradients
        error += bellman_error * features[state, :]

    # Update weights according to the Equation (11.9). Derivatives happen to be the same matrix due to the linearity
    weights += step_size /len(states) * error

    # endregion Body

# endregion Semi-gradient Methods

# region Gradient-TD Methods

def TDC(state, weights, LLS_solution, step_size_w, step_size_v):
    # region Summary
    """
    Temporal-Difference with Gradient Correction (TDC, GTD(0))
    :param state: Current state
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :param LLS_solution: Solution to a linear least-squares (LLS) problem (denoted as 𝒗)
    :param step_size_w: Step-size parameter for 𝒘 (denoted as 𝛼)
    :param step_size_v: Step-size parameter for 𝒗 (denoted as 𝛽)
    :return: Next state
    """
    # endregion Summary

    # region Body

    # Select an action according to the behavior policy


    # Get the next state


    # Calculate the importance sampling ratio (denoted as 𝜌) according to the Equation (11.1)


    # Calculate TD error (denoted as 𝛿) according to the Equation (11.3) for linear case, i.e., 𝑣 ̂(𝑠,𝒘) = 𝒘^𝑇 ∙ 𝒙(𝑠)


    # Update weights according to the sampling equation on page 280


    # Update LLS solution according to the Least Mean Square (LMS) rule (equation on page 279, under the Equation (11.28))




    # endregion Body

def expected_TDC(weights, LLS_solution, step_size_w, step_size_v):
    # region Summary
    """
    Expected Temporal-Difference with Gradient Correction (Expected TDC)
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :param LLS_solution: Solution to a linear least-squares (LLS) problem (denoted as 𝒗)
    :param step_size_w: Step-size parameter for 𝒘 (denoted as 𝛼)
    :param step_size_v: Step-size parameter for 𝒗 (denoted as 𝛽)
    """
    # endregion Summary

    # region Body

    # For every state

        # when computing expected update target, if next state is not lower state, importance sampling ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state


        # calculate the importance sampling ratio


        # under behavior policy, state distribution is uniform, so the probability for each state is 1.0 / len(states)


        # update weights


        # calculate expected update for LLS solution


        # update LLS solution


    # endregion Body

# endregion Gradient-TD Methods

# region Emphatic-TD Methods

def expected_emphatic_TD(weights, emphasis, step_size):
    # region Summary
    """
    Expected Emphatic Temporal-Difference (Expected ETD) which performs synchronous update for both weights and emphasis
    :param weights: Weight for each element of the feature vector (denoted as 𝒘)
    :param emphasis: Emphasis (denoted as 𝑀)
    :param step_size: Step-size parameter (denoted as 𝛼)
    :return: Expected next emphasis
    """
    # endregion Summary

    # region Body

    # Initialize expected update to 0


    # Initialize expected next emphasis to 0.0


    # For every state

        # calculate the importance sampling ratio (denoted as 𝜌)


        # update emphasis (3rd equation on page 282)


        # when computing expected update target, if next state is not lower state, importance sampling ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state


        # calculate expected update


    # update weights




    # endregion Body

# endregion Emphatic-TD Methods

# endregion Functions
