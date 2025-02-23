from state import get_all_states
from player import RLPlayer, HumanPlayer
from judge import Judge

# Get all possible board configurations
all_states = get_all_states(rows=3, columns=3)

# region Functions

def train(epochs: int, print_every_n: int = 500):
    # region Summary
    """
    Train 2 RL players
    :param epochs: number of epochs for training
    :param print_every_n: number of epochs to print the intermediate win rate
    """
    # endregion Summary

    # region Body

    # Create 2 RL players with ε = 0.01 exploring probability


    # Create a judge to organize the game


    # Set the initial win rate of both players to 0


    # For every epoch

        # get the winner


        # check which player is the winner


        # print the intermediate win rates, if needed


        # update value estimates of both players


        # reset the judge => players


    # Save the players' policies


    # endregion Body


def compete(turns):
    # region Summary
    """
    Compete trained RL players
    :param turns: number of turns for competition
    """
    # endregion Summary

    # region Body

    # Create 2 RL players with ε = 0 exploring probability (i.e. greedy)


    # Create a judge to organize the game


    # Load the players' policies


    # Set the initial win rate of both players to 0


    # For every turn

        # get the winner


        # check which player is the winner


        # reset the judge => players




    # endregion Body


def play():
    # region Summary
    """
    Play against RL player. The game is a 0-sum game. If both players are playing with an optimal strategy, every game will end in a tie.
    So we test whether the RL can guarantee at least a tie if it plays 2nd.
    """
    # endregion Summary

    # region Body

    while True:
        # Create a human player


        # Create RL player with ε = 0 exploring probability (i.e. greedy)


        # Create a judge to organize the game


        # Load the RL player's policy


        # Get the winner


        # Check which player is the winner


    # endregion Body

# endregion Functions


if __name__ == '__main__':
    train(epochs=int(1e5))
    compete(turns=int(1e3))
    play()
