import numpy as np
from copy import deepcopy
import heapq

class TrivialModel:
    # region Summary
    """
    Trivial model for planning in Dyna-Q
    """
    # endregion Summary

    # region Constructor

    def __init__(self, rand=np.random):
        # region Summary
        """
        Constructor of TrivialModel class
        :param rand: An instance of np.random.RandomState for sampling
        """
        # endregion Summary

        # region Body

        self.model = dict()
        self.rand = rand

        # endregion Body

    # endregion Constructor

    # region Functions

    def feed(self, state, action, next_state, reward):
        # region Summary
        """
        Feed the model with previous experience
        :param state: Current state
        :param action: Action
        :param next_state: Next state
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        # If the current state is new to the model
        if tuple(state) not in self.model.keys():
            # create a "key=state : value=empty dict" pair in the model
            self.model[tuple(state)] = dict()

        # Set the next state and reward as model's value for current state and action
        self.model[tuple(state)][action] = [list(next_state), reward]

        # endregion Body

    def sample(self):
        # region Summary
        """
        Randomly sample from previous experience
        :return: Current state, action, next state, reward
        """
        # endregion Summary

        # region Body

        # Get the state index
        state_index = self.rand.choice(range(len(self.model.keys())))

        # Get the state by state index
        state = list(self.model)[state_index]

        # Get the action index
        action_index = self.rand.choice(range(len(self.model[state].keys())))

        # Get the action by action index
        action = list(self.model[state])[action_index]

        # Get the next state and reward
        next_state, reward = self.model[state][action]

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward

        # endregion Body

    # endregion Functions

class TimeModel:
    # region Summary
    """
    Time-based model for planning in Dyna-Q+
    """
    # endregion Summary

    # region Constructor

    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        # region Summary
        """
        Constructor of TimeModel class
        :param maze: The maze instance. Indeed, it's not very reasonable to give access to maze to the model.
        :param time_weight: The weight for elapsed time in sampling reward (denoted as κ (kappa), it needs to be small)
        :param rand: An instance of np.random.RandomState for sampling
        """
        # endregion Summary

        # region Body

        self.maze = maze
        self.time_weight = time_weight
        self.rand = rand

        self.model = dict()

        # Track the total time
        self.time = 0

        # endregion Body

    # endregion Constructor

    # region Functions

    def feed(self, state, action, next_state, reward):
        # region Summary
        """
        Feed the model with previous experience
        :param state: Current State
        :param action: Action
        :param next_state: Next state
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        # Increment time
        self.time += 1

        # If the current state is new to the model
        if tuple(state) not in self.model.keys():
            # create a "key=state : value=empty dict" pair in the model
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions.values():
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of 0․ Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(state)][action_] = [list(state), 0, 1]

        # Set the next state, reward and time as model's value for current state and action
        self.model[tuple(state)][action] = [list(next_state), reward, self.time]

        # endregion Body

    def sample(self):
        # region Summary
        """
        Randomly sample from previous experience
        :return: Current state, action, next state, reward
        """
        # endregion Summary

        # region Body

        # Get the state index
        state_index = self.rand.choice(range(len(self.model.keys())))

        # Get the state by state index
        state = list(self.model)[state_index]

        # Get the action index
        action_index = self.rand.choice(range(len(self.model[state].keys())))

        # Get the action by action index
        action = list(self.model[state])[action_index]

        # Get the next state, reward and time
        next_state, reward, time = self.model[state][action]

        # Adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward

        # endregion Body

    # endregion Functions

class PriorityQueue:
    # region Constructor

    def __init__(self):
        # Create an empty list for priority queue
        self.pq = []

        # Create an empty dict for entry finder
        self.entry_finder = {}

        # Create a flag for removed task
        self.REMOVED = '<removed-task>'

        # Initialize a counter with 0
        self.counter = 0

    # endregion Constructor

    # region Functions

    def remove_item(self, item):
        # region Summary
        """
        Remove item
        :param item: An item to remove
        """
        # endregion Summary

        # region Body

        # Pop an item from entry finder
        entry = self.entry_finder.pop(item)

        # Set the item as removed
        entry[-1] = self.REMOVED

        # endregion Body

    def add_item(self, item, priority=0):
        # region Summary
        """
        Add item
        :param item: An item to add
        :param priority: Priority
        """
        # endregion Summary

        # region Body

        # Check if item is in entry finder
        if item in self.entry_finder:
            self.remove_item(item)

        # Form an entry
        entry = [priority, self.counter, item]

        # Increment counter
        self.counter += 1

        # Add the entry to entry finder
        self.entry_finder[item] = entry

        # Push the entry to priority queue
        heapq.heappush(self.pq, entry)

        # endregion Body

    def pop_item(self):
        # region Summary
        """
        Pop item.
        :return: Item, priority
        """
        # endregion Summary

        # region Body

        while self.pq:
            priority, count, item = heapq.heappop(self.pq)

            if item is not self.REMOVED:
                del self.entry_finder[item]

                return item, priority

        raise KeyError("Pop from an empty priority queue")

        # endregion Body

    def empty(self):
        # region Summary
        """
        Check whether the entry finder is empty
        :return: True, if entry finder is empty; otherwise, False
        """
        # endregion Summary

        # region Body

        return not self.entry_finder

        # endregion Body

    # endregion Functions

class PriorityModel(TrivialModel):
    # region Summary
    """
    Model containing a priority queue for Prioritized Sweeping
    """
    # endregion Summary

    # region Constructor

    def __init__(self, rand=np.random):
        # region Summary
        """
        Constructor of PriorityModel class
        :param rand: An instance of np.random.RandomState for sampling
        """
        # endregion Summary

        # region Body

        # Base class constructor call
        TrivialModel.__init__(self, rand)

        # Maintain a priority queue
        self.priority_queue = PriorityQueue()

        # Track predecessors for every state
        self.predecessors = dict()

        # endregion Body

    # endregion Constructor

    # region Functions

    def insert(self, priority, state, action):
        # region Summary
        """
        Add a given state-action pair into the priority queue with given priority.
        :param priority: Priority
        :param state: State
        :param action: Action
        """
        # endregion Summary

        # region Body

        # Note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((tuple(state), action), -priority)

        # endregion Body

    def empty(self):
        # region Summary
        """
        Check whether the priority queue is empty
        :return: True, if the priority queue is empty; otherwise, False
        """
        # endregion Summary

        # region Body

        return self.priority_queue.empty()

        # endregion Body

    def sample(self):
        # region Summary
        """
        Get the 1st item in the priority queue
        :return: Priority with opposite sign, current state, action, next state, reward
        """
        # endregion Summary

        # region Body

        # Get the current state, action and priority
        (state, action), priority = self.priority_queue.pop_item()

        # Get the next state and reward
        next_state, reward = self.model[state][action]

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        return -priority, list(state), action, list(next_state), reward

        # endregion Body

    def feed(self, state, action, next_state, reward):
        # region Summary
        """
        Feed the model with previous experience
        :param state: Current state
        :param action: Action
        :param next_state: Next state
        :param reward: Reward
        """
        # endregion Summary

        # region Body

        # Deep copy the current state
        state = deepcopy(state)

        # Deep copy the next state
        next_state = deepcopy(next_state)

        # Feed the TrivialModel with previous experience
        TrivialModel.feed(self, state, action, next_state, reward)

        # If the next state hasn't been in the predecessors
        if tuple(next_state) not in self.predecessors.keys():
            # create a "key=next_state : value=empty set" pair in the predecessors
            self.predecessors[tuple(next_state)] = set()

        # Add the current state and action to the predecessors' next state
        self.predecessors[tuple(next_state)].add((tuple(state), action))

        # endregion Body

    def predecessor(self, state):
        # region Summary
        """
        Get all seen predecessors of given state.
        :param state: State
        :return: All seen predecessors of given state
        """
        # endregion Summary

        # region Body

        # If the state hasn't been in the predecessors, return an empty list
        if tuple(state) not in self.predecessors.keys():
            return []

        # Create an empty list for predecessors
        predecessors = []

        # Get all predecessors of given state
        for state_predecessors, action_predecessors in list(self.predecessors[tuple(state)]):
            predecessors.append([list(state_predecessors), action_predecessors, self.model[state_predecessors][action_predecessors][1]])

        return predecessors

        # endregion Body

    # endregion Functions
