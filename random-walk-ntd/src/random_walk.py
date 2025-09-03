import numpy as np

# region Hyper-parameters

# Number of states
states_number = 19

# Discount rate (denoted as 𝛾)
discount = 1

# Non-terminal states
non_terminal_states = np.arange(1, states_number + 1)

# Terminal states. An action leading to the:
#                                           left terminal state has reward -1,
#                                           right terminal state has reward 1.
terminal_states = [0, states_number + 1]

# True state-values of non-terminal states from Bellman equation
true_state_values = np.arange(-20, 22, 2) / 20.0

# True state-values of terminal states
true_state_values[0] = true_state_values[-1] = 0

# Start from the middle state
start = 10

# endregion Hyper-parameters

# region Functions

def temporal_difference(state_value_estimates, steps_number, step_size):
    # region Summary
    """
    n-steps TD Method
    :param state_value_estimates: Value estimates for each state (denoted as V)
    :param steps_number: Number of steps (denoted as n)
    :param step_size: Step-size parameter (denoted as 𝛼)
    """
    # endregion Summary

    # region Body

    # Initialize starting state
    current_state = start

    # List to store states for an episode
    states = [current_state]

    # List to store rewards for an episode
    rewards = [0]

    # Track the time
    time_step = 0

    # Define the length of this episode
    termination_time = float('inf')

    while True:
        # Move to next time step
        time_step += 1

        # If the episode is not over
        if time_step < termination_time:
            # choose an action randomly
            if np.random.binomial(n = 1, p = 0.5) == 1:
                next_state = current_state + 1
            else:
                next_state = current_state - 1


            # check the left terminal state
            if next_state == 0:
                reward = -1

            # check the right terminal state
            elif next_state == 20:
                reward = 1

            # reward for all non-terminal states is 0
            else:
                reward = 0

            # store the new state
            states.append(next_state)

            # store the new reward
            rewards.append(reward)

            # check terminal states
            if next_state in terminal_states:
                termination_time = time_step

        # Get the time of the state to update
        update_time = time_step - steps_number

        if update_time >= 0:
            estimated_return = 0.0

            # Calculate corresponding returns (Equation (7.1)):
            for t in range(update_time + 1, min(termination_time, update_time + steps_number) + 1):
                estimated_return += pow(discount, t - update_time - 1) * rewards[t]

            # If episode is not over
            if update_time + steps_number <= termination_time:

                # add state-value estimate to the return (Equation (7.1)):
                estimated_return += (pow(discount, steps_number) *
                                     state_value_estimates[states[(update_time + steps_number)]])

            # Get the state to update
            state_to_update = states[update_time]

            # If state to be updated is a non-terminal state
            if not state_to_update in terminal_states:
                # update the state-value estimate (Equation (7.2)):
                state_value_estimates[state_to_update] += (step_size *
                                        (estimated_return - state_value_estimates[state_to_update]))

        # If episode is over
        if update_time == termination_time - 1:
            break

        # Move to the next state
        current_state = next_state

    # endregion Body

# endregion Functions
