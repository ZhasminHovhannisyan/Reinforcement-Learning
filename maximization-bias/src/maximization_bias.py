import numpy as np
import random

# region Hyper-parameters

# States
states = dict(A = 0, B = 1, terminal = 2)

# Start state
start = states['A']

# Possible actions from state A
actions_A = dict(right = 0, left = 1)

# Possible actions from state B (e.g., 10 actions (will affect the curves))
actions_B = range(0, 10)

# All possible actions
actions = [[actions_A["right"], actions_A["left"]], actions_B]

# State-action pair values. The value of a terminal state is always 0
state_action_values = [np.zeros(2), np.zeros(len(actions_B)), np.zeros(1)]

# Destination for each state and each action
transition = [[states["terminal"], states['B']], [states["terminal"]] * len(actions_B)]

# Exploration probability (denoted as 𝜀)
exploration_probability = 0.1

# Step-size parameter (denoted as 𝛼)
step_size = 0.1

# Discount rate for max value (denoted as 𝛾)
discount = 1.0

# endregion Hyper-parameters

# region Functions

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
        action = np.random.choice(actions[state])

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    else:
        values = action_value_estimates[state]
        action = np.random.choice([act for act, val in enumerate(values) if val == np.max(values)])

    return action

    # endregion Body

def take_action(state, action):
    # region Summary
    """
    Takes an action in state and gets reward
    :param state: State
    :param action: Action
    :return: Reward
    """
    # endregion Summary

    # region Body

    # The reward from state A is 0 regardless of the action
    if state == states['A']:
        return 0

    # The reward from state B is drawn from a normal distribution with 𝜇 = -0.1 mean and 𝜎 = 1.0 variance for all possible actions
    return np.random.normal(-0.1, 1)

    # endregion Body

def q_learning(first_action_value_estimates, second_action_value_estimates = None):
    # region Summary
    """
    Counts the number of "left" action for Q-learning or Double Q-learning
    :param first_action_value_estimates: 1st action-value estimates (denoted as 𝑄_1(𝑎))
    :param second_action_value_estimates: 2nd action-value estimates (denoted as 𝑄_2(𝑎)). If not None, then this function is Double Q-learning; otherwise, it is classic Q-learning
    :return: Number of "left" action in state A
    """
    # endregion Summary

    # region Body

    # Initialize state at the start
    state = start

    # Track the number of action "left" in state A
    left_count = 0

    # Keep going until getting to the terminal state
    while state != states["terminal"]:

        # choose an action for classic Q-learning
        if second_action_value_estimates is None:
            action = choose_action(action_value_estimates=first_action_value_estimates, state = state)

        # choose an action for Double Q-learning
        else:

            # for example, an 𝜀-greedy policy for Double Q-learning could be based on the average (or sum) of the 2 action-value estimates
            action = choose_action(
                action_value_estimates=(
                    [first_estimate + second_estimate for first_estimate, second_estimate in
                     zip(first_action_value_estimates, second_action_value_estimates)]
                ),
                state = state
            )

        # check if agent chose "left" action in state A
        if action == actions_A["left"] and state == states['A']:
            left_count += 1

        # get the reward
        reward = take_action(state, action)

        # get the next state
        next_state = transition[state][action]

        # for classic Q-learning
        if second_action_value_estimates is None:

            # set action-value estimate to update
            update_estimate = first_action_value_estimates

            # set target
            target = np.max(update_estimate[next_state])


        # for Double Q-learning, divide the time steps in 2, perhaps by flipping a coin on each step
        else:

            # if the coin comes up heads
            if np.random.binomial(n=1, p=0.5):

                # set the estimate to update to 𝑄_1
                update_estimate = first_action_value_estimates

                # set the target estimate to 𝑄_2
                target_estimate = second_action_value_estimates

            # if the coin comes up tails, then the same update is done with 𝑄_1 and 𝑄_2 switched, so that 𝑄_2 is updated
            else:

                # set the estimate to update to 𝑄_2
                update_estimate = second_action_value_estimates

                # set the target estimate to 𝑄_1
                target_estimate = first_action_value_estimates


            # get the best action
            best_action = random.choice(
                [act for act, val in enumerate(update_estimate[next_state])
                 if val == np.max(update_estimate[next_state])]
            )

            # get the target
            target = target_estimate[next_state][best_action]

        # Q-learning update (Equation (6.8))
        update_estimate[state][action] += step_size * (reward + discount * target - update_estimate[state][action])

        # move to the next state
        state = next_state

    return left_count

    # endregion Body

def expected_sarsa(first_action_value_estimates):
    # region Summary
    """
    One episode of Expected SARSA. Returns max_Q_B (for computing bias).
    """
    # endregion Summary

    #region Body

    # Initialize state at the start
    state = start

    # Keep going until getting to the terminal state
    while state != states["terminal"]:

        # Choose action: ε-greedy selection from Q
        if np.random.binomial(n=1, p=exploration_probability) == 1:

            # With probability ε, pick a random action
            action = np.random.choice(actions[state])

        # Otherwise, pick one of the greedy actions (highest Q-value)
        else:
            values = first_action_value_estimates[state]
            greedy = [act for act, v in enumerate(values) if v == np.max(values)]
            action = random.choice(greedy)

        # Take the action in the environment and observe reward and next_state
        reward = take_action(state, action)
        next_state = transition[state][action]
        done = (next_state == states["terminal"])

        # if not "terminal", calculate expected Q-value in next_state under ε-greedy polic
        if not done:
            action_list = actions[next_state]
            N = len(action_list)

            # Find all greedy actions in next_state (highest Q among candidates)
            q_vals_next = first_action_value_estimates[next_state]
            max_q_next = np.max(q_vals_next)
            greedy_next = [a2 for a2, v in enumerate(q_vals_next) if v == max_q_next]

            # Probability of selecting any action uniformly = ε / N
            p_uniform = exploration_probability / N

            # Total probability for greedy actions = (1 - ε)
            p_greedy_total = 1 - exploration_probability

            # Compute the expected Q-value: sum over all a2 of π(a2|next_state) * Q(next_state, a2)
            exp_q = 0.0
            for a2 in action_list:
                if a2 in greedy_next:
                    # Greedy action probability = (ε/N) + (1-ε)/number_of_greedy
                    exp_q += (p_uniform + p_greedy_total / len(greedy_next)) * first_action_value_estimates[next_state][a2]

                else:
                    # Non-greedy action probability = ε/N
                    exp_q += p_uniform * first_action_value_estimates[next_state][a2]

            # Update Q(s, a): Q ← Q + α [r + γ * expected_Q_next - Q]
            first_action_value_estimates[state][action] += \
                step_size * (reward + discount * exp_q - first_action_value_estimates[state][action])

            # Move to next_state for next iteration
            state = next_state

        else:
            # next_state == terminal → target = reward + γ * 0
            first_action_value_estimates[state][action] += \
                step_size * (reward + discount * 0 - first_action_value_estimates[state][action])
            break

    # At the end of the episode, return max Q-value in state B (for bias calculation)
    return np.max(first_action_value_estimates[states["B"]])

    # endregion Body

def double_expected_sarsa(Q1, Q2):
    # region Summary
    """
    One episode of Double Expected SARSA. Returns max_Q_B over the average (Q1+Q2)/2 at the end.
    """
    # endregion Summary

    # region Body

    # Initialize state at the start of the episode
    state = start

    # Continue until reaching the terminal state
    while state != states["terminal"]:
        # Construct ε-greedy policy using the sum Q1 + Q2 for the current state
        combined = [q1 + q2 for q1, q2 in zip(Q1[state], Q2[state])]

        # With probability ε, select a random action
        if np.random.binomial(n=1, p=exploration_probability) == 1:
            action = np.random.choice(actions[state])

        # Otherwise, choose one of the greedy actions (highest combined Q)
        else:
            greedy = [act for act, v in enumerate(combined) if v == np.max(combined)]
            action = random.choice(greedy)

        # Take the action in the environment and observe reward and next_state
        reward = take_action(state, action)
        next_state = transition[state][action]
        done = (next_state == states["terminal"])

        # Randomly decide which table to update: Q1 or Q2
        if np.random.binomial(n=1, p=0.5):
            update_Q = Q1
            target_Q = Q2
        else:
            update_Q = Q2
            target_Q = Q1

        # If next_state ≠ terminal, compute the expected Q-value under ε-greedy on (Q1+Q2)
        if not done:
            action_list = actions[next_state]
            N = len(action_list)

            # Find all greedy actions in next_state based on Q1 + Q2
            q_sum_next = [Q1[next_state][a2] + Q2[next_state][a2] for a2 in action_list]
            max_q_sum = np.max(q_sum_next)
            greedy_next = [a2 for a2, v in zip(action_list, q_sum_next) if v == max_q_sum]

            # Probability of selecting any action uniformly = ε / N
            p_uniform = exploration_probability / N

            # Total probability for greedy actions = (1 - ε)
            p_greedy_total = 1 - exploration_probability

            # Compute the expected Q-value: sum over a2 of π(a2|next_state) * target_Q(next_state, a2)
            exp_q = 0.0
            for a2 in action_list:
                if a2 in greedy_next:
                    # Greedy action probability = (ε/N) + (1 - ε)/number_of_greedy
                    exp_q += (p_uniform + p_greedy_total / len(greedy_next)) * target_Q[next_state][a2]
                else:
                    # Non-greedy action probability = ε/N
                    exp_q += p_uniform * target_Q[next_state][a2]

            # Update the chosen Q-table: Q ← Q + α [r + γ * expected_Q_next - Q]
            update_Q[state][action] += step_size * (reward + discount * exp_q - update_Q[state][action])

            # Move to next_state for the next iteration
            state = next_state

        else:
            # next_state == terminal → target = reward + γ * 0
            update_Q[state][action] += step_size * (reward + discount * 0 - update_Q[state][action])
            break

    # At the end of the episode, return max over the average (Q1+Q2)/2 in state B
    avg_B = [(Q1[states["B"]][a] + Q2[states["B"]][a]) / 2 for a in range(len(actions[states["B"]]))]
    return np.max(avg_B)

    # endregion Body


# endregion Functions