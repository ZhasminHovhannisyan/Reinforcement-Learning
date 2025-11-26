import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# region Hyper-parameters

# Number of states
states_number = 19

# All non-terminal states
states = np.arange(1, states_number + 1)

# Start from the middle state
start = 10

# 2 terminal states: an action leading to the:
# 1. left terminal state has reward -1,
# 2. right terminal state has reward 1
ends = [0, states_number + 1]

# True state-values from Bellman equation
true_values = np.arange(-20, 22, 2) / 20.0
true_values[0] = true_values[states_number + 1] = 0.0

# endregion Hyper-parameters

# region VF Classes

class ValueFunction:
    # region Summary
    """
    Base class for 𝜆-based algorithms in Eligibility Traces (ETs)
    In this example, we use:
    1. the simplest linear feature function: state aggregation,
    2. exact 19 groups, so the weights for each group is exact the value for that state.
    """
    # endregion Summary

    # region Constructor

    def __init__(self, trace_decay, step_size):
        # region Summary
        """
        Constructor of VF class
        :param trace_decay: Trace-decay parameter (denoted as 𝜆)
        :param step_size: Step-size parameter (denoted as 𝛼)
        """
        # endregion Summary

        # region Body

        self.trace_decay = trace_decay
        self.step_size = step_size

        # Initialize weights vector (denoted as 𝒘)
        self.weights = np.zeros(states_number + 2)

        # endregion Body

    # endregion Constructor

    # region Functions

    def value(self, state):
        # region Summary
        """
        Returns the value of given state.
        :param state: State
        :return: State-value
        """
        # endregion Summary

        # region Body

        # The state-value is just the weight
        return self.weights[state]

        # endregion Body

    def initialize_episode(self):
        # region Summary
        """
        Initialize some variables at the beginning of each episode.
        Must be called at the very beginning of each episode. Derived class should override this function
        """
        # endregion Summary

        # region Body

        return

        # endregion Body

    def learn(self, state, reward):
        # region Summary
        """
        Feed the algorithm with new observation. Derived class should override this function
        :param state: State
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        return

        # endregion Body

    # endregion Functions

class OffLineLambdaReturn(ValueFunction):
    # region Summary
    """
    Off-line 𝜆-return algorithm
    """
    # endregion Summary

    # region Constructor

    def __init__(self, trace_decay, step_size):
        # region Summary
        """
        Constructor of OffLineLambdaReturn class
        :param trace_decay: Trace-decay parameter (denoted as 𝜆)
        :param step_size: Step-size parameter (denoted as 𝛼)
        """
        # endregion Summary

        # region Body

        # Call the constructor of base class
        ValueFunction.__init__(self, trace_decay, step_size)

        # To accelerate learning, set a truncate value for power of 𝜆
        self.trace_decay_truncate = 1e-3

        # endregion Body

    # endregion Constructor

    # region Functions

    def n_step_return_from_time(self, steps_number, time):
        # region Summary
        """
        Get the n-step return from the given time (𝛾 = 1 always and rewards = 0 except for the last reward => Equation (12.1) can be simplified)
        :param steps_number: Number of steps (denoted as n)
        :param time: Time step (denoted as t)
        :return: n-step return (denoted as 𝐺_(𝑡:𝑡+𝑛))
        """
        # endregion Summary

        # region Body

        # Calculate end time
        end_time = min(time + steps_number, self.episode_length)

        # Calculate n-step return
        n_step_return = self.value(self.trajectory[end_time])

        # If episode ended
        if end_time == self.episode_length:
            # add reward to n-step return
            n_step_return += self.reward

        return n_step_return

        # endregion Body

    def lambda_return_from_time(self, time):
        # region Summary
        """
        Get the 𝜆-return from the given time.
        :param time: Time step (denoted as t)
        :return: 𝜆-return (denoted as 𝐺_𝑡^𝜆)
        """
        # endregion Summary

        # region Body

        # Initialize 𝜆-return
        lambda_returns = 0.0

        # Set 𝜆's power
        lambda_power = 1

        # For n=1 to n=T-t-1 (Equation (12.3))
        for n in range(1, self.episode_length - time):
            # calculate 𝜆-return (1st additive of the equation)
            lambda_returns += lambda_power * self.n_step_return_from_time(n, time)

            # raise 𝜆's power
            lambda_power *= self.trace_decay

            # if the power of 𝜆 has been too small,
            if lambda_power < self.trace_decay_truncate:
                # discard all the following sequences
                break

        # Multiply 𝜆-return by (1 - 𝜆)
        lambda_returns *= (1 - self.trace_decay)

        # If 𝜆's power is greater than or equal to 𝜆's truncate value
        if lambda_power >= self.trace_decay_truncate:
            # add the post-termination terms (2nd additive of the equation) to 𝜆-return
            lambda_returns += lambda_power * self.reward

        return lambda_returns

        # endregion Body

    def off_line_learn(self):
        # region Summary
        """
        Perform off-line learning at the end of an episode
        """
        # endregion Summary

        # region Body

        # For every time step
        for time_step in range(self.episode_length):
            # get state
            state = self.trajectory[time_step]

            # update weights according to the Equation (12.4) for every state in the trajectory
            self.weights[state] += self.step_size * (self.lambda_return_from_time(time_step) - self.value(state))

        # endregion Body

    def initialize_episode(self):
        # region Summary
        """
        Initialize some variables at the beginning of each episode.
        Must be called at the very beginning of each episode
        """
        # endregion Summary

        # region Body

        # Initialize the trajectory
        self.trajectory = [start]

        # Only need to track the last reward in one episode, as all others are 0
        self.reward = 0.0

        # endregion Body

    def learn(self, state, reward):
        # region Summary
        """
        Feed the algorithm with new observation
        :param state: State
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Add the new state to the trajectory
        self.trajectory.append(state)

        # If state is terminal
        if state in ends:
            # save the reward
            self.reward = reward

            # calculate the length of episode (denoted as T)
            self.episode_length = len(self.trajectory) - 1

            # start off-line learning once the episode ends
            self.off_line_learn()

        # endregion Body

    # endregion Functions

class TemporalDifferenceLambda(ValueFunction):
    # region Summary
    """
    TD(𝜆) algorithm
    """
    # endregion Summary

    # region Constructor

    def __init__(self, trace_decay, step_size):
        # region Summary
        """
        Constructor of TemporalDifferenceLambda class
        :param trace_decay: Trace-decay parameter (denoted as 𝜆)
        :param step_size: Step-size parameter (denoted as 𝛼)
        """
        # endregion Summary

        # region Body

        # Call the constructor of base class
        ValueFunction.__init__(self, trace_decay, step_size)

        # Initialize some variables at the beginning of each episode
        self.initialize_episode()

        # endregion Body

    # endregion Constructor

    # region Functions

    def initialize_episode(self):
        # region Summary
        """
        Initialize some variables at the beginning of each episode.
        Must be called at the very beginning of each episode
        """
        # endregion Summary

        # region Body

        # Initialize the ET vector (denoted as 𝒛)
        self.accumulating_trace = np.zeros(states_number + 2)

        # Initialize the beginning state
        self.last_state = start

        # endregion Body

    def learn(self, state, reward):
        # region Summary
        """
        Feed the algorithm with new observation
        :param state: State
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Update the ET vector (Equation (12.5))
        self.accumulating_trace *= self.trace_decay
        self.accumulating_trace[self.last_state] += 1

        # Calculate TD error (denoted as 𝛿, Equation (12.6))
        TD_error = reward + self.value(state) - self.value(self.last_state)

        # Update weights vector (Equation (12.7))
        self.weights += self.step_size * TD_error * self.accumulating_trace

        # Update the last state
        self.last_state = state

        # endregion Body

    # endregion Functions

class OnLineLambdaReturn(ValueFunction):
    # region Summary
    """
    Online 𝜆-return algorithm = True online TD(𝜆) algorithm
    """
    # endregion Summary

    # region Constructor

    def __init__(self, trace_decay, step_size):
        # region Summary
        """
        Constructor of OnLineLambdaReturn class
        :param trace_decay: Trace-decay parameter (denoted as 𝜆)
        :param step_size: Step-size parameter (denoted as 𝛼)
        """
        # endregion Summary

        # region Body

        # Call the constructor of base class
        ValueFunction.__init__(self, trace_decay, step_size)

        # endregion Body

    # endregion Constructor

    # region Functions

    def initialize_episode(self):
        # region Summary
        """
        Initialize some variables at the beginning of each episode.
        Must be called at the very beginning of each episode
        """
        # endregion Summary

        # region Body

        # Initialize the ET vector (denoted as 𝒛)
        self.dutch_trace = np.zeros(states_number + 2)

        # Initialize the beginning state
        self.last_state = start

        # Initialize the old state-value
        self.old_state_value = 0.0

        # endregion Body

    def learn(self, state, reward):
        # region Summary
        """
        Feed the algorithm with new observation
        :param state: State
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Get last state's value
        last_state_value = self.value(self.last_state)

        # Get current state's value
        state_value = self.value(state)

        # Update the ET vector (Equation (12.11))
        self.dutch_trace *= self.trace_decay
        self.dutch_trace[self.last_state] += 1 - self.step_size * self.trace_decay * self.dutch_trace[self.last_state]

        # Calculate TD error (denoted as 𝛿, Equation (12.6))
        TD_error = reward + state_value - last_state_value

        # Update weights vector (equation on page 300)
        self.weights += self.step_size * (TD_error + last_state_value - self.old_state_value) * self.dutch_trace
        self.weights[self.last_state] -= self.step_size * (last_state_value - self.old_state_value)

        # Update old state-value
        self.old_state_value = state_value

        # Update the last state
        self.last_state = state

        # endregion Body

    # endregion Functions

# endregion VF Classes

# region Functions

def random_walk(value_function):
    # region Summary
    """
    19-state random walk
    :param value_function: Instance of VF or derived class
    """
    # endregion Summary

    # region Body

    # Initialize episode
    value_function.initialize_episode()

    # Start state
    state = start

    # While episode doesn't end
    while state not in ends:
        # choose an action
        action = np.random.choice([-1, 1])

        # get the next state
        next_state = state + action

        # left terminal state
        if next_state == 0:
            reward = -1

        # right terminal state
        elif next_state == states_number + 1:
            reward = 1

        # other states
        else:
            reward = 0

        # feed the VF with new observation
        value_function.learn(next_state, reward)

        # move to the next state
        state = next_state

    # endregion Body

def parameter_sweep(value_function_generator, runs, trace_decays, step_sizes, title):
    # region Summary
    """
    General plot framework
    :param value_function_generator: Generates an instance of VF
    :param runs: Number of independent runs
    :param trace_decays: Series of different trace-decay parameter (𝜆) values
    :param step_sizes: Sequences of step-size parameter (𝛼) for each 𝜆
    :param title: Plot title
    """
    # endregion Summary

    # region Body

    # Play for 10 episodes for each run
    episodes = 10

    # Track the RMSEs
    rms_errors = [np.zeros(len(ss)) for ss in step_sizes]

    # For every run
    for _ in tqdm(range(runs)):
        # for every trace-decay parameter
        for lambda_index, trace_decay in enumerate(trace_decays):
            # for every step-size parameter
            for alpha_index, step_size in enumerate(step_sizes[lambda_index]):
                # generate an instance of VF
                value_function = value_function_generator(trace_decay, step_size)

                # for every episode
                for episode in range(episodes):
                    # perform 19-state random walk
                    random_walk(value_function)

                    # get estimated state values
                    state_value_estimates = [value_function.value(state) for state in states]

                    # calculate RMSE
                    rms_errors[lambda_index][alpha_index] += np.sqrt(np.mean(np.power(state_value_estimates - true_values[1: -1], 2)))

    # Average over runs and episodes
    for error in rms_errors:
        error /= episodes * runs

    # Plotting
    for i in range(len(trace_decays)):
        plt.plot(step_sizes[i], rms_errors[i], label=fr"$\lambda = {trace_decays[i]}$")

    plt.title(title)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(f"RMS error at the end of the episode over the first {episodes} episodes")
    plt.legend()

    # endregion Body

# endregion Functions
