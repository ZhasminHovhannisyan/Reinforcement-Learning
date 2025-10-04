import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


# region Hyper-parameters

# 2 possible actions
actions = [0, 1]

# Each transition has a probability to terminate with 0
termination_probability = 0.1

# Maximum expected updates
max_steps = 20000

# Exploration probability for behavior policy
exploration_probability = 0.1

# endregion Hyper-parameters

# region Functions

# region Helpers

def argmax(action_value_estimates):
    # region Summary
    """
    Break tie randomly when selecting action with maximum Q-value
    :param action_value_estimates: Action-value estimates
    :return: Action with maximum estimated value
    """
    # endregion Summary

    # region Body

    # Get the maximum of action-value estimates
    max_estimate = np.max(action_value_estimates)

    # Get the action with maximum action-value estimate
    action = np.random.choice([action_ for action_, action_value_estimate in enumerate(action_value_estimates) if action_value_estimate == max_estimate])

    return action

    # endregion Body

class Task:
    # region Constructor

    def __init__(self, n_states, branching_factor):
        # region Summary
        """
        MDP Task with random transitions and rewards
        :param n_states: Number of non-terminal states
        :param branching_factor: Branching factor (number of possible next states per state-action pair)
        """
        # endregion Summary

        # region Body

        self.n_states = n_states
        self.branching_factor = branching_factor

        # Pre-compute transition matrix and rewards for all state-action pairs
        self.transition = np.random.randint(n_states, size=(n_states, len(actions), branching_factor))
        self.reward = np.random.randn(n_states, len(actions), branching_factor)

        # Pre-compute termination mask for vectorized operations
        self.termination_mask = np.random.rand(max_steps) < termination_probability
        self.termination_idx = 0

        # Pre-compute random choices for next state selection
        self.random_choices = np.random.randint(branching_factor, size=max_steps)
        self.choice_idx = 0

        # endregion Body

    # endregion Constructor

    # region Functions

    def step(self, state, action):
        # region Summary
        """
        Take a step in the environment
        :param state: Current state
        :param action: Action
        :return: Next state (n_states if terminal) and reward for this transition
        """
        # endregion Summary

        # region Body

        # Check termination using pre-computed values
        if self.termination_mask[self.termination_idx % max_steps]:
            self.termination_idx += 1
            return self.n_states, 0.0

        # Select next state using pre-computed random choice
        next_idx = self.random_choices[self.choice_idx % max_steps]
        self.choice_idx += 1
        self.termination_idx += 1

        # Get the next state
        next_state = self.transition[state, action, next_idx]

        # Get the reward
        reward = self.reward[state, action, next_idx]

        return next_state, reward

        # endregion Body

    # endregion Functions

def evaluate_policy(state_action_value_estimates, task, runs=1000):
    # region Summary
    """
    Evaluate the value of start state under greedy policy using Monte Carlo
    :param state_action_value_estimates: State-action value estimates
    :param task: MDP task
    :param runs: Number of MC runs
    :return: Average return from start state
    """
    # endregion Summary

    # region Body

    # Create an array for returns filled with 0s
    returns = np.zeros(runs)

    # For every run
    for run in range(runs):
        # initialize total reward
        total_reward = 0.0

        # track the number of states
        state = 0

        # track the number of steps
        steps = 0

# while there are states in task and step limit is not exceeded
        while state < task.n_states and steps < 1000:
            # get the action with maximum estimated value
            action = argmax(state_action_value_estimates[state])

            # get the next state and reward
            next_state, reward = task.step(state, action)

            # add the reward to total rewards
            total_reward += reward

            # update state for next iteration
            state = next_state

            # increment the number of steps
            steps += 1

        # add the total reward to the returns for current run
        returns[run] = total_reward

    return np.mean(returns)

    # endregion Body

# endregion Helpers

# region Sampling

def uniform_sampling(task, eval_interval):
    # region Summary
    """
    Perform expected updates with uniform state-action distribution
    :param task: MDP task
    :param eval_interval: Steps between evaluations
    :return: Tuple of (steps, performance) arrays
    """
    # endregion Summary

    # region Body

    # Create an empty list for performance
    performance = []

    # Create a matrix for state-action value estimates filled with 0s
    estimates = np.zeros((task.n_states, len(actions)))

    # Pre-compute evaluation steps
    eval_steps = np.arange(0, max_steps, eval_interval, dtype=int)
    eval_idx = 0

    # For every step
    for step in tqdm(range(max_steps), desc = 'Uniform Sampling'):

        # get the current state
        state = (step // len(actions)) % task.n_states

        # get the action
        action = step % len(actions)

        # get the next states
        next_states = task.transition[state, action]

        # get the rewards
        rewards = task.reward[state, action]

        # get the next value estimates
        next_value_estimates = np.max(estimates[next_states], axis=1)

        # update estimates
        estimates[state, action] = (1-termination_probability) * np.mean(rewards + next_value_estimates)

        # check if it's time to evaluate
        if eval_idx < len(eval_steps) and step == eval_steps[eval_idx]:
            # get the average return
            average_return = evaluate_policy(estimates, task)

            # append the step and average return to the performance list
            performance.append([step, average_return])

            # increment the evaluation index
            eval_idx += 1

    return list(zip(*performance))

    # endregion Body


def on_policy_sampling(task, eval_interval):
    # region Summary
    """
    Perform expected updates with on-policy distribution
    :param task: MDP task
    :param eval_interval: Steps between evaluations
    :return: Tuple of (steps, performance) arrays
    """
    # endregion Summary

    # region Body

    # Create an empty list for performance
    performance = []

    # Create a matrix for state-action value estimates filled with 0s
    estimates = np.zeros((task.n_states, len(actions)))

    # Initialize state counter to 0
    state = 0

    # Pre-compute evaluation steps
    eval_steps = np.arange(0, max_steps, eval_interval, dtype=int)
    eval_idx = 0

    # Pre-compute ε-greedy decisions
    epsilon_decisions = np.random.rand(max_steps) < exploration_probability

    # Get random actions
    random_actions = np.random.choice(actions, max_steps)

    # For every step
    for step in tqdm(range(max_steps), desc = 'On-Policy Sampling'):

        # ε-greedy action selection
        if epsilon_decisions[step]:
            # choose random action
            action = random_actions[step]
        else:
            # choose action with maximum estimated value
            action = argmax(estimates[state])

        # get the next state
        next_state, _ = task.step(state, action)

        # get the next states
        next_states = task.transition[state, action]

        # get the rewards
        rewards = task.reward[state, action]

        # get the next value estimates
        next_value_estimates = np.max(estimates[next_states], axis=1)

        # update estimates
        estimates[state, action] = (1 - termination_probability) * np.mean(rewards + next_value_estimates)

        # reset to start state if terminal
        state = 0 if next_state == task.n_states else next_state

        # check if it's time to evaluate
        if eval_idx < len(eval_steps) and step == eval_steps[eval_idx]:
            # get the average return
            average_return = evaluate_policy(estimates, task)

            # append the step and average return to the performance list
            performance.append([step, average_return])

            # increment the evaluation index
            eval_idx += 1


    return list(zip(*performance))


    # endregion Body

    # endregion Sampling

    # region Runs

def run_single_task(args):
    # region Summary
    """
    Run a single task for parallel processing
    :param args: Tuple of (task, method, eval_interval)
    :return: Performance data for this task
    """
    # endregion Summary

    # region Body

    task, method, eval_interval = args

    return method(task, eval_interval)

    # endregion Body


def run_experiment(n_states, branching_factor, methods, n_tasks=30, evaluation_points=100):
    # region Summary
    """
    Run the complete experiment for given parameters
    :param n_states: Number of states
    :param branching_factor: Branching factor
    :param methods: List of methods to test
    :param n_tasks: Number of tasks to average over
    :param evaluation_points: Number of evaluation points
    :return: Dictionary with results for each method
    """
    # endregion Summary

    # region Body

    # Get the evaluation interval
    eval_interval = max_steps // evaluation_points

    # Create an empty dictionary for results
    results = {}

    # Generate all tasks upfront
    tasks = [Task(n_states, branching_factor) for _ in range(n_tasks)]

    # For every method
    for method in methods:
        print(f"Running {method.__name__} for {n_states} states, b={branching_factor}")

        # prepare arguments for parallel processing
        args_list = [(task, method, eval_interval) for task in tasks]

        # use parallel processing for better performance
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            task_results = list(executor.map(run_single_task, args_list))

        # aggregate results
        all_steps = None
        all_values = []

        # get the task results
        for steps, values in task_results:
            if all_steps is None:
                all_steps = np.array(steps)
            all_values.append(values)

        # calculate mean performance across tasks
        mean_values = np.mean(np.array(all_values), axis=0)
        results[method.__name__] = (all_steps, mean_values)

    return results

    # endregion Body


# endregion Runs

# endregion Functions
