import numpy as np
from tqdm import tqdm

from src.models import TrivialModel, TimeModel

def choose_action(state, action_value_estimates, maze, dyna_params):
    # region Summary
    """
    Choose an action based on 𝜀-greedy algorithm
    :param state: State
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param maze: Maze instance containing all information about the environment
    :param dyna_params: Parameters of Dyna algorithms
    :return: Action
    """
    # endregion Summary

    # region Body

    # ε-greedy action selection: every once in a while, with small probability ε, select randomly from among all the actions with equal probability, independently of the action-value estimates.
    if np.random.binomial(n=1, p=dyna_params.exploration_probability) == 1:
        action = np.random.choice(list(maze.actions.values()))

    # Greedy action selection: select one of the actions with the highest estimated value, that is, one of the greedy actions.
    # If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly.
    else:
        values = action_value_estimates[state[0], state[1], :]
        action = np.random.choice([act for act, val in enumerate(values) if val == np.max(values)])

    return action

    # endregion Body

# region Dyna Maze

def dyna_q(action_value_estimates, model, maze, dyna_params):
    # region Summary
    """
    Play for an episode for Dyna-Q algorithm
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param model: Model instance for planning
    :param maze: Maze instance containing all information about the environment
    :param dyna_params: Parameters of Dyna algorithms
    :return: Steps
    """
    # endregion Summary

    # region Body

    # Start at the maze's start state
    state = maze.start

    # Initialize a counter for steps to 0
    steps = 0

    # While the agent hasn't reached its goal
    while state not in maze.goals:

        # track the steps
        steps+=1

        # choose an action
        action = choose_action(state, action_value_estimates, maze, dyna_params)

        # get the next state and reward
        next_state, reward = maze.step(state, action)

        # perform Q-Learning update
        action_value_estimates[state[0], state[1], action] += (dyna_params.step_size
                    * (reward + dyna_params.discount * np.max(action_value_estimates[next_state[0], next_state[1], :])
                    - action_value_estimates[state[0], state[1], action]))

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            action_value_estimates[state_[0], state_[1], action_] += (dyna_params.step_size *
                    (reward_ + dyna_params.discount * np.max(action_value_estimates[next_state_[0], next_state_[1], :])
                    - action_value_estimates[state_[0], state_[1], action_]))

        # move to next state
        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break
    return steps

    # endregion Body

# endregion Dyna Maze

# region Changing Maze

def changing_maze(maze, dyna_params):
    # region Summary
    """
    Wrapper function for changing maze.
    :param maze: Maze instance containing all information about the environment
    :param dyna_params: Parameters of Dyna algorithms
    :return: Rewards
    """
    # endregion Summary

    # region Body

    # Set up max steps
    max_steps = maze.max_steps

    # Track the cumulative rewards
    rewards = np.zeros((dyna_params.runs, 2, max_steps))

    # For every run
    for run in tqdm(range(dyna_params.runs)):

        # set up models
        models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]

        # initialize state-action value estimates with 0s
        action_value_estimates = [np.zeros(maze.action_value_estimates_size), np.zeros(maze.action_value_estimates_size)]

        # for every method
        for i in range(len(dyna_params.methods)):

            # print('run:', run, dyna_params.methods[i])

            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles

            # initialize a counter for steps to 0
            steps = 0

            # get the last steps
            last_steps = steps

            # while the max steps hasn't been reached
            while steps < max_steps:

                # play for an episode
                steps += dyna_q(action_value_estimates[i], models[i], maze, dyna_params)

                # update cumulative rewards
                rewards[run, i, last_steps:steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1

                # get the last steps
                last_steps = steps

                # if it's time to change the obstacles
                if steps > maze.obstacle_switch_time:
                    # change the obstacles
                    maze.obstacles = maze.new_obstacles

    # Average rewards
    rewards = rewards.mean(axis=0)

    return rewards

    # endregion Body

# endregion Changing Maze

# region Prioritized Sweeping

def prioritized_sweeping(action_value_estimates, model, maze, dyna_params):
    # region Summary
    """
    Play for an episode for prioritized sweeping algorithm
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param model: Model instance for planning
    :param maze: Maze instance containing all information about the environment
    :param dyna_params: Parameters of Dyna algorithms
    :return: Number of backups during this episode
    """
    # endregion Summary

    # region Body

    # Start at the maze's start state
    state = maze.start

    # Track the number of steps in this episode
    steps = 0

    # Track the number of backups in planning phase
    backups = 0

    # While the agent hasn't reached its goal
    while state not in maze.goals:

        # increment the number of steps
        steps += 1

        # choose an action
        action = choose_action(state, action_value_estimates, maze, dyna_params)

        # get the next state and reward
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # get the priority for current state-action pair
        priority = np.abs(reward
                          + dyna_params.discount * np.max(action_value_estimates[next_state[0], next_state[1], :])
                          - action_value_estimates[state[0], state[1], action])

        # check whether priority exceeds threshold
        if priority > dyna_params.threshold:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps (although, keep planning until the priority queue becomes empty will converge much faster)
        while planning_step < dyna_params.planning_steps and not model.empty():
            # get a sample with the highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            delta = (reward_ + dyna_params.discount
                     * np.max(action_value_estimates[next_state_[0], next_state_[1], :])
                     - action_value_estimates[state_[0], state_[1], action_])

            # update action-value estimates
            action_value_estimates[state_[0], state_[1], action_] += dyna_params.step_size * delta

            # deal with all the predecessors of the sample state
            # Fix: Convert state_ to tuple for dictionary key lookup
            state_key = tuple(state_)
            if state_key in model.predecessors:
                for state_predecessors, action_predecessors in model.predecessors[state_key]:
                    # Get the reward from the model for this state-action pair
                    reward_pre = model.model[tuple(state_predecessors)][action_predecessors][1]

                    priority = np.abs(reward_pre + dyna_params.discount
                                      * np.max(action_value_estimates[state_[0], state_[1], :])
                                      - action_value_estimates[
                                          state_predecessors[0], state_predecessors[1], action_predecessors])

                    # check whether priority exceeds threshold
                    if priority > dyna_params.threshold:
                        model.insert(priority, state_predecessors, action_predecessors)

            # increment the planning step
            planning_step += 1

        # move to the next state
        state = next_state

        # update the number of backups
        backups += planning_step + 1

    return backups

    # endregion Body
def check_path(action_value_estimates, maze):
    # region Summary
    """
    Check whether state-action values are already optimal
    :param action_value_estimates: Action-value estimates (denoted as 𝑄(𝑆_𝑡, 𝐴_𝑡))
    :param maze: Maze instance containing all information about the environment
    :return: True, if state-action values are already optimal; otherwise, False
    """
    # endregion Summary

    # region Body

    # Set the length of optimal path of the original maze
    original_length = 14

    # Relaxed optimal path
    relaxed_optimal_path = 1.2

    # Get the length of optimal path
    max_steps = original_length * maze.resolution * relaxed_optimal_path

    # Start at the maze's start state
    state = maze.start

    # Track the number of steps in this episode
    steps = 0

    # While the agent hasn't reached its goal
    while state not in maze.goals:
        # get the action with maximum estimated value
        action = np.argmax(action_value_estimates[state[0], state[1], :])

        # get the state
        state, _ = maze.step(state, action)

        # increment the number of steps
        steps += 1

        # check whether the number of steps exceeds maximum
        if steps > max_steps:
            return False

    return True

    # endregion Body

# endregion Prioritized Sweeping