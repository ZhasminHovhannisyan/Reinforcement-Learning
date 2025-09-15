class Maze:
    # region Summary
    """
    A wrapper class for a maze, containing all the information about the maze.
    By default, it's initialized to DynaMaze. However, it can be easily adapted to other maze.
    """
    # endregion Summary

    # region Constructor

    def __init__(self):
        # Maze width and height
        self.world = dict(width = 9, height = 6)

        # All possible actions
        self.actions = dict(up = 0, down = 1, left = 2, right = 3)

        # Start state
        self.start = [2, 0]

        # Goal states
        self.goals = [[0, 8]]

        # All obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

        # Old obstacles
        self.old_obstacles = None

        # New obstacles
        self.new_obstacles = None

        # Time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # The size of action-value estimates
        self.action_value_estimates_size = (self.world["height"], self.world["width"], len(self.actions))

        # Max steps
        self.max_steps = float('inf')

        # Track the resolution for this maze
        self.resolution = 1

    # endregion Constructor

    # region Functions

    def extend_state(self, state, factor):
        # region Summary
        """
        Extend a state into higher resolution
        :param state: State in lower resolution maze
        :param factor: Extension factor (one state will become factor^2 states after extension)
        :return: New states
        """
        # endregion Summary

        # region Body

        # Get the new state by multiplying current state with factor
        new_state = [state[0] * factor, state[1] * factor]

        # Create an empty list for new states
        new_states = []

        for i in range(factor):
            for j in range(factor):
                new_states.append([new_state[0] + i, new_state[1] + j])

        return new_states

        # endregion Body

    def extend_maze(self, factor):
        # region Summary
        """
        Extend a state to a higher resolution maze
        :param factor: Extension factor (one state in original maze will become factor^2 states in new maze)
        :return: New maze
        """
        # endregion Summary

        # region Body

        # Create a new maze
        new_maze = Maze()

        # Set the width of the new maze
        new_maze.world["width"] = self.world["width"] * factor

        # Set the height of the new maze
        new_maze.world["height"] = self.world["height"] * factor

        # Set the start state of the new maze
        new_maze.start = [self.start[0] * factor, self.start[1] * factor]

        # Set the goal states of the new maze
        new_maze.goals = self.extend_state(self.goals[0], factor)

        # Set the obstacles of the new maze
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))

        # Set the action-value estimates size of the new maze
        new_maze.action_value_estimates_size = (new_maze.world["height"], new_maze.world["width"], len(new_maze.actions))

        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))

        # Set the resolution of the new maze
        new_maze.resolution = factor

        return new_maze

        # endregion Body

    def step(self, state, action):
        # region Summary
        """
        Take action in state.
        :param state: State
        :param action: Action
        :return: New state and reward
        """
        # endregion Summary

        # region Body

        # Get the state's coordinates
        x, y = state

        # Check if action is "up"
        if action == self.actions["up"]:
            x = max(x - 1, 0)

        # Check if action is "down"
        elif action == self.actions["down"]:
            x = min(x + 1, self.world["height"] - 1)

        # Check if action is "left"
        elif action == self.actions["left"]:
            y = max(y - 1, 0)

        # Check if action is "right"
        elif action == self.actions["right"]:
            y = min(y + 1, self.world["width"] - 1)

        # If the agent's movement is blocked by an obstacle,
        if [x, y] in self.obstacles:
            # the agent remains where it is
            x, y = state

        # If the agent reached its goal,
        if [x, y] in self.goals:
            # reward is +1
            reward = 1.0
        else:
            # otherwise, reward is 0 on all transitions
            reward = 0.0

        return [x, y], reward

        # endregion Body

    # endregion Functions
