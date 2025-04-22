import numpy as np

# region Fields

# Actions
actions = dict(left = 0, right = 1)

# endregion Fields

# region Functions

def target_policy():
    # region Summary
    """
    Target Policy
    :return: Action
    """
    # endregion Summary

    # region Body

    # The target policy always selects 'left' (i.e. 𝜋(left | 𝑠) = 1)
    return actions["left"]

    # endregion Body

def behavior_policy():
    # region Summary
    """
    Behavior Policy
    :return: Action
    """
    # endregion Summary

    # region Body

    # The behavior policy selects 'right' and 'left' with equal probability, i.e. b(left | s) = 1/2 = b(right | s)
    return np.random.binomial(n=1, p=0.5)

    # endregion Body

def play():
    # region Summary
    """
    Play
    :return: Reward, trajectory of actions
    """
    # endregion Summary

    # region Body

    # Track the actions for importance sampling ratio
    actions_trajectory = []

    while True:
        # Choose an action according to the behavior policy
        action = behavior_policy()

        # Append the action to the trajectory
        actions_trajectory.append(action)

        # If action is 'right', the reward is 0
        if action == actions["right"]:
            return 0, actions_trajectory

        # If action is 'left', the reward is 1 if environment transitions on to terminal state
        if np.random.binomial(n=1, p=0.9) == 0:
            return 1, actions_trajectory

    # endregion Body

# endregion Functions
