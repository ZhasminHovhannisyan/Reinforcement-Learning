import numpy as np

class State:
    # region Constructor

    def __init__(self, rows: int = 3, columns: int = 3):
        # region Summary
        """
        State of the game (denoted as 𝑆)
        :param rows: number of board's rows
        :param columns: number of board's columns

        The Tic-Tac-Toe game board is represented by an n * n array, where:
        a. 0 represents an empty position,
        b. 1 represents the player who moves first,
        c. -1 represents another player.
        """
        # endregion Summary

        # region Body

        self.board_rows = rows
        self.board_columns = columns
        self.board_size = self.board_rows * self.board_columns

        # At the beginning of the game, the board is empty <=> it is filled with 0s
        self.data = np.zeros((self.board_rows, self.board_columns))

        # Player that won the game
        self.winner = None

        # Unique hash value for state
        self.hash_value = None

        # True, if the game ended; otherwise, false
        self.game_ended = None

        # endregion Body

    # endregion Constructor

    # region Functions

    def calculate_hash_value(self):
        # region Summary
        """
        Calculates a unique hash value for state.
        :return: hash value
        """
        # endregion Summary

        # region Body

        if self.hash_value is None:
            self.hash_value = 0
            for i in np.nditer(self.data):
                self.hash_value = self.hash_value * 3 + i + 1
        return self.hash_value

        # endregion Body

    def is_game_ended(self):
        # region Summary
        """
        Checks whether a player has won the game, or it's a tie.
        :return: True, if the game ended; otherwise, False
        """
        # endregion Summary

        # region Body

        if self.game_ended is not None:
            return self.game_ended

        results = []

        # Sum rows
        for i in range(self.board_rows):
            results.append(np.sum(self.data[i, :]))

        # Sum columns
        for i in range(self.board_columns):
            results.append(np.sum(self.data[:, i]))

        # Sum diagonals
        main_diagonal = 0
        secondary_diagonal = 0
        for i in range(self.board_rows):
            main_diagonal += self.data[i, i]
            secondary_diagonal += self.data[i, self.board_rows - 1 - i]
        results.append(main_diagonal)
        results.append(secondary_diagonal)

        # Check win
        for result in results:
            if result == 3:
                self.winner = 1
                self.game_ended = True
                return self.game_ended
            if result == -3:
                self.winner = -1
                self.game_ended = True
                return self.game_ended

        # Check tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == self.board_size:
            self.winner = 0
            self.game_ended = True
            return self.game_ended

        # Otherwise, the game is still on
        self.game_ended = False
        return self.game_ended

        # endregion Body

    def get_next_state(self, i, j, symbol):
        # region Summary
        """
        Get the next state by putting player's symbol in position (i, j).
        :param i: position's row number
        :param j: position's column number
        :param symbol: 1 or -1
        :return: next state
        """
        # endregion Summary

        # region Body

        # Create the next state
        next_state = State(self.board_rows, self.board_columns)

        # Copy the previous state's data into the next state
        next_state.data = np.copy(self.data)

        # Change the symbol at (i, j) position
        next_state.data[i, j] = symbol

        return next_state

        # endregion Body

    def print_state(self):
        # region Summary
        """
        Print the current state of the board
        """
        # endregion Summary

        # region Body

        for i in range(self.board_rows):
            print('-------------')
            out = '| '
            for j in range(self.board_columns):
                if self.data[i, j] == 1:
                    symbol = '*'
                elif self.data[i, j] == -1:
                    symbol = 'x'
                else:
                    symbol = '0'
                out += symbol + ' | '
            print(out)
        print('-------------')

        # endregion Body

    # endregion Functions


def _get_all_states(current_state, current_symbol, all_states, rows: int = 3, columns: int = 3):
    # region Summary
    """
    Private function for getting all states
    :param current_state: Current state
    :param current_symbol: Current symbol
    :param all_states: Dictionary of all states
    :param rows: Number of board's rows
    :param columns: Number of board's columns
    """
    # endregion Summary

    # region Body

    for i in range(rows):
        for j in range(columns):
            if current_state.data[i][j] == 0:
                new_state = current_state.get_next_state(i, j, current_symbol)
                new_hash = new_state.calculate_hash_value()
                if new_hash not in all_states:
                    is_end = new_state.is_game_ended()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        _get_all_states(new_state, -current_symbol, all_states, rows, columns)

    # endregion Body


def get_all_states(rows: int = 3, columns: int = 3):
    # region Summary
    """
    Public function for getting all states
    :param rows: Number of board's rows
    :param columns: Number of board's columns
    :return: dictionary of all states
    """
    # endregion Summary

    # region Body

    current_symbol = 1
    current_state = State(rows, columns)
    all_states = dict()
    all_states[current_state.calculate_hash_value()] = (current_state, current_state.is_game_ended())
    _get_all_states(current_state, current_symbol, all_states, rows, columns)
    return all_states

    # endregion Body
