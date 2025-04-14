import numpy as np
from tqdm import tqdm

# region Fields

# region For Player

# region Actions

# If the player does not have a natural, then he can request additional cards, one by one (hits), ->
hit = 0

# -> until he either stops (sticks) or exceeds 21 (goes bust).
stick = 1

# The player’s actions are to hit or to stick.
actions = [hit, stick]

# endregion Actions

# region Policy

# Policy for player
player_policy = np.zeros(22, dtype=np.int64)

for i in range(12, 20):
    player_policy[i] = hit

player_policy[20] = stick
player_policy[21] = stick

# endregion Policy

# endregion For Player

# region For Dealer

# region Policy

# Policy for dealer
dealer_policy = np.zeros(22)

for i in range(12, 17):
    dealer_policy[i] = hit

for i in range(17, 22):
    dealer_policy[i] = stick

# endregion Policy

# endregion For Dealer

# endregion Fields

# region Functions

# region Helper Functions

# region Policies

def target_policy_player(usable_ace_player, player_sum, dealer_card):
    # region Summary
    """
    Target policy of player
    :param usable_ace_player: Whether a player has usable ace
    :param player_sum: Sum of player's cards
    :param dealer_card: Dealer's showing card
    :return: Action
    """
    # endregion Summary

    # region Body

    return player_policy[player_sum]

    # endregion Body

def behavior_off_policy_player(usable_ace_player, player_sum, dealer_card):
    # region Summary
    """
    Behavior policy of player for Off-policy Monte Carlo Sampling
    :param usable_ace_player: Whether a player has usable ace
    :param player_sum: Sum of player's cards
    :param dealer_card: Dealer's showing card
    :return: Action
    """
    # endregion Summary

    # region Body

    if np.random.binomial(n=1, p=0.5) == 1:
        return stick
    return hit

    # endregion Body

# endregion Policies

# region Cards

def get_card():
    # region Summary
    """
    Get a new card.
    :return: Card
    """
    # endregion Summary

    # region Body

    card = np.random.randint(low=1, high=14)
    card = min(card, 10)
    return card

    # endregion Body

def card_value(card_id):
    # region Summary
    """
    Get the value of a card.
    :param card_id: Card's ID, i.e. number on card (ace = 1, number cards = respective number, all face cards = 10)
    :return: Card's value (11 for ace, card's ID for the rest)
    """
    #endregion Summary

    # region Body

    return 11 if card_id == 1 else card_id

    # endregion Body

def update_cards(card_sum, usable_ace):
    # region Summary
    """
    Update sum of cards and count of aces for player or dealer
    :param card_sum: Sum of cards
    :param usable_ace: Whether ace is usable
    :return: Updated sum of cards and count of aces
    """
    # endregion Summary

    # region Body

    # Get a new card
    new_card = get_card()

    # Keep track of the ace count. The usable_ace flag is insufficient alone as it cannot distinguish between having one 1 or 2 aces.
    ace_count = int(usable_ace)

    # If the new card is ace, increment the ace count
    if new_card == 1:
        ace_count += 1

    # Add the new card's value to the sum of cards
    card_sum += card_value(new_card)

    # If the player has a usable ace, use it as 1 to avoid busting and continue.
    while card_sum > 21 and ace_count:
        card_sum -= 10
        ace_count -= 1

    return card_sum, ace_count

    # endregion Body

# endregion Cards

# endregion Helper Functions

def play(policy_player, initial_state=None, initial_action=None):
    # region Summary
    """
    Play a game
    :param policy_player: Specify policy for player
    :param initial_state: Whether player has a usable ace, sum of player's cards, 1 card of dealer
    :param initial_action: The initial action
    :return: State, reward, player trajectory
    """
    # endregion Summary

    # region Body

    # region Player

    # Sum of player's cards
    player_sum = 0

    # Trajectory of player
    player_trajectory = []

    # Whether player uses ace as 11
    usable_ace_player = False

    # If no initial state is given, generate a random initial state:
    if initial_state is None:
        # while sum of player's cards is less than 12, always choose hit as action:
        while player_sum < 12:
            # get a new card
            card = get_card()

            # add card's value to the sum of player's cards
            player_sum += card_value(card)

            # If the sum of player's cards is larger than 21, # he may hold 1 or 2 aces.
            if player_sum > 21:
                assert player_sum == 22

                # last card must be ace
                player_sum -= 10
            else:
                # track if the player has at least one usable ace (counted as 11). Operator |= acts as a logical OR assignment.
                # usable_ace_player remains True if already set; becomes True if the new card is an ace (1). Example:
                # 1. if usable_ace_player = False and card = 1 (ace) => usable_ace_player becomes True,
                # 2. if usable_ace_player = True (already has a usable ace) => usable_ace_player stays True regardless of new cards.
                usable_ace_player |= (1 == card)

        # Initialize cards of dealer
        dealer_card1 = get_card() # suppose dealer will show the 1st card he gets
        dealer_card2 = get_card()

    else: # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # State of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # endregion Player

    # region Dealer

    # Sum of dealer's cards
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)

    # Whether dealer has a usable ace
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)

    # If the sum dealer's cards is greater than 21, he must hold 2 aces
    if dealer_sum > 21:
        assert dealer_sum == 22

        # use one ace as 1 rather than 11
        dealer_sum -= 10

    # endregion Dealer

    assert dealer_sum <= 21
    assert player_sum <= 21

    # region Game

    # Player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # Get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # Track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        # If player sticks, then it becomes the dealer’s turn
        if action == stick:
            break

        # If player hits, update player's cards
        player_sum, ace_count = update_cards(player_sum, usable_ace_player)

        # Check if player busts
        if player_sum > 21:
            return state, -1, player_trajectory

        # If player doesn't bust, then he can have a usable ace if he has only 1 ace
        usable_ace_player = (ace_count == 1)

    # Dealer's turn
    while True:
        # Get action based on current sum
        action = dealer_policy[dealer_sum]

        # If dealer sticks, then it is time define the winner
        if action == stick:
            break

        # If dealer hits, update dealer's cards
        dealer_sum, ace_count = update_cards(dealer_sum, usable_ace_dealer)

        # Check if dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory

        # If dealer doesn't bust, then he can have a usable ace if he has only 1 ace
        usable_ace_dealer = (ace_count == 1)

    # endregion Game

    # region Winner

    # Compare the sum of cards between player and dealer
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

    # endregion Winner

    # endregion Body

# region Monte Carlo Functions

def monte_carlo_on_policy(episodes):
    # region Summary
    """
    Monte Carlo On-Policy Sampling
    :param episodes: Number of episodes
    :return: Sample-average of returns of states with usable ace, sample-average of returns of states with no usable ace
    """
    # endregion Summary

    # region Body

    # States with usable ace (i.e. player holds an ace that he could count as 11 without going bust)
    states_usable_ace = np.zeros((10,10))

    # Initialize counts with 1s in order to avoid division by 0
    states_usable_ace_count = np.ones((10,10))

    # States with no usable ace
    states_no_usable_ace = np.zeros((10,10))

    # Initialize counts with 1s in order to avoid division by 0
    states_no_usable_ace_count = np.ones((10,10))

    # For every episode
    for _ in tqdm(range(episodes)):

        # get reward and player's trajectory
        _, reward, player_trajectory = play(target_policy_player)

        # for every triple in player's trajectory
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:

            # adjust the sum of player's cards (10 values: 12–21 → 0–9)
            player_sum -= 12

            #adjust the ID of dealer's card (10 values: 1–10 → 0–9)
            dealer_card -= 1

            # check if player has a usable ace
            if usable_ace:
                states_usable_ace[player_sum, dealer_card] += reward
                states_usable_ace_count[player_sum, dealer_card] += 1
            else:
                states_no_usable_ace[player_sum, dealer_card] += reward
                states_no_usable_ace_count[player_sum, dealer_card] += 1

    # Calculate average of states with usable ace
    average_usable_ace = states_usable_ace / states_usable_ace_count

    # Calculate average of states with no usable ace
    average_no_usable_ace = states_no_usable_ace / states_no_usable_ace_count

    return average_usable_ace, average_no_usable_ace

    # endregion Body

def monte_carlo_es(episodes):
    # region Summary
    """
    Monte Carlo with Exploring Starts (ES)
    :param episodes: Number of episodes
    :return: Sample-average of state-action values
    """
    # endregion Summary

    # region Body

    # Create a 4-dimensional tensor for state-action values with dimensions = (player_sum, dealer_card, usable_ace, action)
    state_action_values = np.zeros((10,10,2,2))

    # Initialize counts with 1s in order to avoid division by 0
    state_action_pair_count = np.ones((10,10,2,2))

    # region Local Functions

    def behavior_policy(usable_ace, player_sum, dealer_card):
        # region Summary
        """
        Behavior policy is greedy
        :param usable_ace: Whether a player has a usable ace
        :param player_sum: Sum of player's cards
        :param dealer_card: Dealer's showing card
        :return: Action
        """
        # endregion Summary

        # region Body

        # Convert usable ace from bool to int
        usable_ace = int(usable_ace)

        # Adjust the sum of player's cards (10 values: 12–21 → 0–9)
        player_sum -= 12

        # Adjust the ID of dealer's card (10 values: 1–10 → 0–9)
        dealer_card -= 1

        # Get argmax of the average returns (s, a)
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / state_action_pair_count[player_sum, dealer_card, usable_ace, :]

        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # endregion Body

    # endregion Local Functions

    # For every episode
    for episode in tqdm(range(episodes)):
        # use a randomly initialized state
        initial_state = [
            bool(np.random.choice([0,1])),
            np.random.choice(range(12,22)),
            np.random.choice(range(1, 11)),
        ]

        # use a randomly initialized action
        initial_action = np.random.choice(actions)

        # select the current policy
        current_policy = behavior_policy if episode else target_policy_player

        # get reward and player's trajectory
        _, reward, player_trajectory = play(current_policy, initial_state, initial_action)

        # create an empty set for 1st visits to state-action pairs
        first_visits = set()


        # for every quartet in player's trajectory
        for(usable_ace, player_sum, dealer_card), action in player_trajectory:

            # convert usable ace from bool to int
            usable_ace = int(usable_ace)

            # adjust the sum of player's cards (10 values: 12–21 → 0–9)
            player_sum -= 12

            # adjust the ID of dealer's card (10 values: 1–10 → 0–9)
            dealer_card -= 1

            # form the state-action pair
            state_action = (usable_ace, player_sum, dealer_card, action)

            # if the state-action pair has been visited before, skip it
            if state_action in first_visits:
                continue

            # else, this is the 1st visit to that state-action pair
            first_visits.add(state_action)

            # update values of state-action pairs
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values

    # endregion Body

def monte_carlo_off_policy(episodes):
    # region Summary
    """
    Monte Carlo Off-Policy Sampling
    :param episodes: Number of episodes
    :return: Ordinary importance sampling, weighted importance sampling
    """
    # endregion Summary

    # region Body



    # endregion Body

# endregion Monte Carlo Functions

# endregion Functions
