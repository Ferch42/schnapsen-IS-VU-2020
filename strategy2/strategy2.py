"""
RandomBot -- A simple strategy: enumerates all legal moves, and picks one
uniformly at random.
"""

# Import the API objects
from api import Deck
from api import State, util
import random


class Bot:

    def __init__(self, randomize=True, depth=6):
        """
        :param randomize: Whether to select randomly from moves of equal value (or to select the first always)
        :param depth:
        """
        self.__randomize = randomize
        self.__max_depth = depth


    def value(self, state, depth = 0):
        # type: (State, int) -> tuple[float, tuple[int, int]]
        """
        Return the value of this state and the associated move
        :param state:
        :param depth:
        :return: A tuple containing the value of this state, and the best move for the player currently to move
        """

        if state.finished():
            winner, points = state.winner()
            return (points, None) if winner == 1 else (-points, None)

        if depth == self.__max_depth:
            return heuristic(state)

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a recursive function call so that 'value' will contain the
            # minimax value of 'next_state'
            value, _ = self.value(next_state, depth = depth+1) 

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_value, best_move


    def get_move(self, state):
        # type: (State) -> tuple[int, int]
        """
        Function that gets called every turn. This is where to implement the strategies.
        Be sure to make a legal move. Illegal moves, like giving an index of a card you
        don't own or proposing an illegal mariage, will lose you the game.
        TODO: add some more explanation
        :param State state: An object representing the gamestate. This includes a link to
            the states of all the cards, the trick and the points.
        :return: A tuple of integers or a tuple of an integer and None,
            indicating a move; the first indicates the card played in the trick, the second a
            potential spouse.
        """
        if state.get_phase() == 1:
            # All legal moves
            moves = state.moves()
            chosen_move = moves[0]

            # Check for trump exchange
            for move in moves:
                if move[0] is None:
                    return move
                    
            # Check for marriage
            for move in moves:
                if move[1] is not None:
                    return move

            moves_trump_suit = []

            #Get all trump suit moves available
            for index, move in enumerate(moves):

                if move[0] is not None and Deck.get_suit(move[0]) == state.get_trump_suit():
                    moves_trump_suit.append(move)

            moves_trump_suit = sorted(moves_trump_suit, key= lambda x: x[0])


            # If the opponent has played a card
            if state.get_opponents_played_card() is not None:

                moves_same_suit = []

                # Get all moves of the same suit as the opponent's played card
                for index, move in enumerate(moves):
                    if move[0] is not None and Deck.get_suit(move[0]) == Deck.get_suit(state.get_opponents_played_card()):
                        moves_same_suit.append(move)

                if state.get_opponents_played_card()%5 ==0 or state.get_opponents_played_card()%5==1:
                    for move in moves_same_suit:
                        if move[0]%5<state.get_opponents_played_card()%5:
                            return move
                    if  Deck.get_suit(state.get_opponents_played_card()) != state.get_trump_suit() and len(moves_trump_suit)>0:
                        return moves_trump_suit[-1]


            # Get move with lowest rank available, of any suit

            moves_low_cards = []
            for index, move in enumerate(moves):
                if move[0] is not None and (move[0]%5 ==4 or move[0]%5 ==3 or move[0]%5 ==2):
                    moves_low_cards.append(move)

            if len(moves_low_cards)>0:
                chosen_move = random.choice(moves_low_cards)

            return chosen_move

        else:

            val, move = self.value(state)
            
            return move

def maximizing(state):
    # type: (State) -> bool
    """
    Whether we're the maximizing player (1) or the minimizing player (2).

    :param state:
    :return:
    """
    return state.whose_turn() == 1

def heuristic(state):
    # type: (State) -> float
    """
    Estimate the value of this state: -1.0 is a certain win for player 2, 1.0 is a certain win for player 1

    :param state:
    :return: A heuristic evaluation for the given state (between -1.0 and 1.0)
    """
    return util.ratio_points(state, 1) * 2.0 - 1.0, None

