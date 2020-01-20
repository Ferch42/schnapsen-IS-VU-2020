#!/usr/bin/env python
"""
A basic adaptive bot. This is part of the third worksheet.

"""

from api import State, util
import random, os
from itertools import chain

from sklearn.externals import joblib


from keras.models import load_model

import numpy as np

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/Qnet.h5'

class Bot:


    def __init__(self,model_file=DEFAULT_MODEL):

        # Load the model

        possible_moves = [(None, None)]
        possible_moves += [(x,None) for x in range(20)]
        possible_moves+= [(None,x) for x in range(20)]
        possible_moves += [(2,3),(3,2),(7,8),(8,7),(12,13),(13,12),(17,18),(18,17)]

        possible_moves_dict = dict()

        for i, move in enumerate(possible_moves):
            possible_moves_dict[move] = i

        self.possible_moves = possible_moves
        self.possible_moves_dict = possible_moves_dict
        self.Qnet = load_model(model_file)

    def move_hot_form(self,move):
    
        hot_form = [0 for _ in range(len(self.possible_moves))]
        hot_form[self.possible_moves_dict[move]] =1
        
        return hot_form

    def get_move(self, state):

        state_features = features(state)
        moves = state.moves()
        state_vectors = [state_features+ self.move_hot_form(m) for m in moves]
        Qvalues = self.Qnet.predict(np.array(state_vectors)).flatten()

        return moves[np.argmax(Qvalues)]
        


def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []



    # Add player 1's points to feature set
    p1_points = state.get_points(state.whose_turn()) 
    # Add player 2's points to feature set
    p2_points = state.get_points(util.other(state.whose_turn())) 

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(state.whose_turn())
    # Add plauer 2's pending points to feature set
    p2_pending_points = state.get_pending_points(util.other(state.whose_turn()))
    # Get trump suit
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase() 
    # Add stock size to feature set
    stock_size =  state.get_stock_size()

    # Add leader to feature set
    leader = 1 if state.leader()==state.whose_turn() else 2

    # Add whose turn it is to feature set
    # whose_turn = state.whose_turn()

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

    ################## You do not need to do anything below this line ########################
    '''
    correct the perspective, whether is 1 or 2
    '''

    player_hand = 'P1H'
    player_win = 'P1W'

    other_hand = 'P2H'
    other_win = 'P2W'

    if state.whose_turn() == 2:
        player_hand = 'P2H'
        player_win = 'P2W'

        other_hand = 'P1H'
        other_win = 'P1W'
    
    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U'   else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S'   else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != player_hand else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != other_hand else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != player_win else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != other_win else [0, 0, 0, 0, 0, 1] for card in perspective]

    # Append one-hot encoded perspective to feature_set
    feature_set += list(chain(*perspective))
    

    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points/total_points if total_points > 0 else 0.)
    feature_set.append(p2_points/total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points/total_pending_points if total_pending_points > 0 else 0.)

    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot

    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]

    # Append normalized stock size to feature set
    feature_set.append(stock_size/10)

    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]

    # Append one-hot encoded whose_turn to feature set
    # feature_set += [1, 0] if whose_turn == 1 else [0, 1]

    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    # Return feature set
    return feature_set

