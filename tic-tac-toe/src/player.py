import numpy as np
import pickle

class RLPlayer:
    # region Constructor

    def __init__(self, all_states, step_size=0.1, epsilon=0.1):
        # region Summary
        """
        Reinforcement Learning Player
        :param all_states: dictionary of all states
        :param step_size: (denoted as 𝛼) the step size to update estimations
        :param epsilon: (denoted as ε) the probability to explore
        """
        # endregion Summary

        # region Body

        self.all_states = all_states
        self.step_size = step_size
        self.epsilon = epsilon

        # State value estimations (denoted as 𝑉(𝑆))
        self.state_value_estimations = dict()

        # The states in which the agent appears during the game
        self.acquired_states = []

        # Boolean list indicating whether the action chosen in a given state is greedy or not
        self.greedy = []

        # RL player's symbol
        self.symbol = 0

        # endregion Body

    # endregion Constructor

    # region Functions

    def reset(self):
        # region Summary
        """
        Reset RL player
        """
        # endregion Summary

        # region Body

        self.acquired_states = []
        self.greedy = []

        # endregion Body

    def set_state(self, state):
        # region Summary
        """
        Set state.
        :param state: state
        """
        # endregion Summary

        # region Body

        self.acquired_states.append(state)
        self.greedy.append(True)

        # endregion Body

    def set_symbol(self, symbol):
        # region Summary
        """
        Set RL player's symbol.
        :param symbol: symbol of RL player
        """
        # endregion Summary

        # region Body

        self.symbol = symbol

        for hash_value in self.all_states:
            state, game_ended = self.all_states[hash_value]

            if game_ended:
                # Check RL player's winning
                if state.winner == self.symbol:
                    self.state_value_estimations[hash_value] = 1.0

                # Check tie
                elif state.winner == 0:
                    self.state_value_estimations[hash_value] = 0.5

                # RL player lost
                else:
                    self.state_value_estimations[hash_value] = 0

            # The game is still on
            else:
                self.state_value_estimations[hash_value] = 0.5

        # endregion Body

    def update_state_value_estimates(self):
        # region Summary
        """
        If a greedy action was selected in a given state, update state value estimations according to the equation:
        𝑉(𝑆_𝑡) = 𝑉(𝑆_𝑡) + 𝛼(𝑉(𝑆_(𝑡 + 1)) − 𝑉(𝑆_𝑡)) (temporal-difference learning method)
        """
        # endregion Summary

        # region Body

        states = [state.calculate_hash_value() for state in self.acquired_states]

        for t in reversed(range(len(states) - 1)):
            temporal_difference_error = self.greedy[t] * (self.state_value_estimations[states[t + 1]] - self.state_value_estimations[states[t]])
            self.state_value_estimations[states[t]] += self.step_size * temporal_difference_error

        # endregion Body

    def act(self, rows: int = 3, columns: int = 3):
        # region Summary
        """
        Choose an action based on state
        :param rows: number of board's rows
        :param columns: number of board's columns
        :return: action
        """
        # endregion Summary

        # region Body

        # Get the current state
        current_state = self.acquired_states[-1]

        # Create an empty lists for next states and positions that are possible from the current state
        next_states = []
        next_positions = []

        for i in range(rows):
            for j in range(columns):
                # If the (i, j) cell is empty
                if current_state.data[i, j] == 0:
                    # consider that cell for the next position
                    next_positions.append([i, j])

                    # get the next state given the RL player's symbol was put in position (i, j)
                    next_state = current_state.get_next_state(i, j, self.symbol)

                    # calculate the hash value of acquired state
                    next_states.append(next_state.calculate_hash_value())

        # Exploratory move: select randomly (with small probability ε) from among the non-greedy moves instead of selecting greedy move
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        # Greedy move: select the move that leads to the state with the greatest estimated value <=> the highest estimated probability of winning
        actions = []
        for hash_value, position in zip(next_states, next_positions):
            actions.append((self.state_value_estimations[hash_value], position))

        # Select randomly from one of the actions with equal values
        np.random.shuffle(actions) #???

        # Sort the actions in descending order according to the state value estimations
        actions.sort(key=lambda a: a[0], reverse=True)

        # Select the position with the highest state value estimation
        action = actions[0][1]

        # Append the RL player's symbol to the action
        action.append(self.symbol)

        return action

        # endregion Body

    def save_policy(self):
        # region Summary
        """
        Save policy
        """
        # endregion Summary

        # region Body

        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.state_value_estimations, f)

        # endregion Body

    def load_policy(self):
        # region Summary
        """
        Load policy
        """
        # endregion Summary

        # region Body

        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.state_value_estimations = pickle.load(f)

        # endregion Body

    # endregion Functions


class HumanPlayer:
    # region Constructor

    def __init__(self):
        # region Summary
        """
        Human Player
        Input a number to put a chessman
        | q | w | e |
        | a | s | d |
        | z | x | c |
        """
        # endregion Summary

        # region Body

        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

        # endregion Body

    # endregion Constructor

    # region Functions

    def reset(self):
        pass

    def set_state(self, state):
        # region Summary
        """
        Set state.
        :param state: state
        """
        # endregion Summary

        # region Body

        self.state = state

        # endregion Body

    def set_symbol(self, symbol):
        # region Summary
        """
        Set human player's symbol.
        :param symbol: symbol of human player
        """
        # endregion Summary

        # region Body

        self.symbol = symbol

        # endregion Body

    def act(self, rows: int = 3, columns: int = 3):
        # region Summary
        """
        Make an action based on state
        :param rows: number of board's rows
        :param columns: number of board's columns
        :return: Action
        """
        # endregion Summary

        # region Body

        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // rows
        j = data % columns
        return i, j, self.symbol

        # endregion Body

    # endregion Functions
