from state import State

class Judge:
    # region Constructor

    def __init__(self, player1, player2):
        # region Summary
        """
        The judge of the game
        :param player1: The player who will move first (symbol = 1)
        :param player2: Another player (symbol = -1)
        """
        # endregion Summary

        # region Body

        self.player1 = player1
        self.player2 = player2

        self.current_player = None

        self.player1_symbol = 1
        self.player2_symbol = -1

        self.player1.set_symbol(self.player1_symbol)
        self.player2.set_symbol(self.player2_symbol)

        self.current_state = State()

        # endregion Body

    # endregion Constructor

    # region Functions

    def reset(self):
        # region Summary
        """
        Reset players
        """
        # endregion Summary

        # region Body

        self.player1.reset()
        self.player2.reset()

        # endregion Body

    def alternate(self):
        # region Summary
        """
        Alternate players
        """
        # endregion Summary

        # region Body

        while True:
            yield self.player1
            yield self.player2

        # endregion Body

    def play(self, all_states, print_state: bool = False):
        # region Summary
        """
        Play the game
        :param all_states: dictionary of all states
        :param print_state: if True, print each board during the game
        :return: the winner player, when game ends
        """
        # endregion Summary

        # region Body

        # Get both players
        players = self.alternate()

        # Reset players
        self.reset()

        # Get the current state
        current_state = State()

        # Set the players' state to current state
        self.player1.set_state(current_state)
        self.player2.set_state(current_state)

        # Print the current state, if needed
        if print_state:
            current_state.print_state()

        while True:
            # Get the current player
            player = next(players)

            # Make an action
            i, j, symbol = player.act()

            # Calculate the hash value for the next state
            next_state_hash = current_state.get_next_state(i, j, symbol).calculate_hash_value()

            # Get the current state
            current_state, is_game_ended = all_states[next_state_hash]

            # Set the current state for both players
            self.player1.set_state(current_state)
            self.player2.set_state(current_state)

            # Print the current state, if needed
            if print_state:
                current_state.print_state()

            # Return winner when game ends
            if is_game_ended:
                return current_state.winner

        # endregion Body

    # endregion Functions
