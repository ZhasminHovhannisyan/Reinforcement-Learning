import numpy as np

class Bandit:
    # region Constructor

    def __init__(self, arms_number: int = 10, use_sample_averages: bool = False, epsilon=0., initial_action_value_estimates=0., confidence_level=None,
                 use_gradient: bool = False, step_size=0.1, use_gradient_baseline: bool = False, true_expected_reward=0.):
        # region Summary
        """
        k-armed Bandit.
        :param arms_number: (denoted as k) number of bandit's arms
        :param use_sample_averages: if True, use sample-average method for estimating action values
        :param epsilon: (denoted as ε) probability for exploration in ε-greedy algorithm
        :param initial_action_value_estimates: (denoted as 𝑄_1(𝑎)) initial estimation for each action value
        :param confidence_level: (denoted as 𝑐) if not None, use Upper-Confidence-Bound (UCB) action selection
        :param use_gradient: if True, use Gradient Bandit Algorithm (GBA)
        :param step_size: (denoted as 𝛼) constant step size for updating estimates
        :param use_gradient_baseline: if True, use average reward as baseline for GBA
        :param true_expected_reward: true expected rewards selected from normal (Gaussian) distribution with μ=4 mean and σ=1 variance
        """
        # endregion Summary

        # region Body

        self.k = arms_number
        self.actions = np.arange(self.k)

        # Value of each action is expected or mean reward given that that action is selected (denoted as 𝑞_∗(𝑎))
        self.action_values = None

        # Estimated value of each action (denoted as 𝑄_𝑡(𝑎))
        self.estimated_action_values = None

        # region Action-Value Methods

        # region Sample-average Method

        self.use_sample_averages = use_sample_averages

        # endregion Sample-average Method

        # region Action Selection Methods

        # region ε-greedy

        self.epsilon = epsilon

        # endregion ε-greedy

        # region Optimistic Initial Values

        self.initial_action_value_estimates = initial_action_value_estimates

        # endregion Optimistic Initial Values

        # region UCB

        self.confidence_level = confidence_level

        # Time steps
        self.time = 0

        # Number of times each action has been selected (denoted as 𝑁_𝑡(𝑎))
        self.action_selection_count = None

        # endregion UCB

        # region GBA

        self.use_gradient = use_gradient

        # Probability of taking action 𝑎 at time 𝑡 (denoted as 𝜋_𝑡(𝑎))
        self.action_probability = None

        self.step_size = step_size

        # Average of the rewards up to (but not including) time 𝑡 (denoted as 𝑅̅_𝑡)
        self.average_reward = 0

        self.use_gradient_baseline = use_gradient_baseline

        self.true_expected_reward = true_expected_reward

        # endregion GBA

        # endregion Action Selection Methods

        # endregion Action-Value Methods

        # Optimal action
        self.optimal_action = None

        # endregion Body

    # endregion Constructor

    # region Functions

    def initialize(self):
        # region Summary
        """
        Initialize action parameters
        """
        # endregion Summary

        # region Body

        # Initialize action values according to a normal (Gaussian) distribution with μ=0 mean and σ=1 variance.
        # In case of GBA, add true_expected_reward != 0.
        self.action_values = np.random.randn(self.k) + self.true_expected_reward

        # In case of realistic initial values, initialize estimated action values with 0s.
        # In case of optimistic initial values, add initial_action_value_estimates != 0
        self.estimated_action_values = np.zeros(self.k) + self.initial_action_value_estimates

        # Set time steps to 0
        self.time = 0

        # Initialize number of times each action has been selected to 0 (none of actions has been selected yet)
        self.action_selection_count = np.zeros(self.k)

        # Optimal action is the action with the highest value
        self.optimal_action = np.argmax(self.action_values)

        # endregion Body

    def act(self):
        # region Summary
        """
        Get an action for this bandit.
        :return: Action
        """
        # endregion Summary

        # region Body

        # region ε-greedy

        # ε-greedy action selection: every once in a while, with small probability ε, select randomly from among all the actions with equal probability, independently of the action-value estimates.


        # endregion ε-greedy

        # region Greedy

        # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
        # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.


        # endregion Greedy

        # endregion Body

    def step(self, action):
        # region Summary
        """
        Update estimated action value and return reward for this action.
        :param action: Action
        :return: Reward
        """
        # endregion Summary

        # region Body

        # When a learning method applied to that bandit problem selected action 𝐴_𝑡 at time step 𝑡, the actual reward, 𝑅_𝑡, was selected from
        # a normal (Gaussian) distribution with μ = 𝑞_∗(𝑎) mean and σ = 1 variance
        actual_reward = np.random.randn() + self.action_values[action]

        # Add 1 to time step
        self.time += 1

        # Add 1 to number of times this action has been selected
        self.action_selection_count[action] += 1

        # The average of the rewards can be computed incrementally
        # The Bandit Gradient Algorithm as Stochastic Gradient Ascent
        self.average_reward += (actual_reward - self.average_reward) / self.time

        if self.use_sample_averages: # Update estimated action values using sample-average method
            # Incremental Implementation (Equation 2.3)
            self.estimated_action_values[action] += (actual_reward - self.estimated_action_values[action]) / self.action_selection_count[action]

        elif self.use_gradient: # Update estimated action values using GBA
            one_hot_encoding = np.zeros(self.k)
            one_hot_encoding[action] = 1

            # The average of the rewards can serve as a baseline with which the reward is compared.
            baseline = self.average_reward if self.use_gradient_baseline else 0

            # A natural learning algorithm for soft-max action preferences based on the idea of stochastic gradient ascent:
            # on each step, after selecting action 𝐴_𝑡 and receiving the reward 𝑅_𝑡, the action preferences are updated by Equation 2.12:
            self.estimated_action_values += self.step_size * (actual_reward - baseline) * (one_hot_encoding - self.action_probability)

        else: # Update estimated action values with constant step size
            # Incremental Implementation (Equation 2.3) with constant step size parameter
            self.estimated_action_values[action] += self.step_size * (actual_reward - self.estimated_action_values[action])

        return actual_reward

        # endregion Body

    # endregion Functions
