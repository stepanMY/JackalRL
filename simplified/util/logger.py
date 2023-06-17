import numpy as np
from copy import deepcopy


def prepare_game_tuple(game):
    """
    Prepares game state tuple: (masked_field, full_field, gold_field,
                      positions, turn_count, gold_left,
                      (first_gold, game.second_gold))

    :param game: object of SimpleGame, example of the game that will be used to calculate constants
    :return:
    """
    progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                      deepcopy(game.positions), game.turn_count, game.gold_left,
                      (game.first_gold, game.second_gold))
    return progress_tuple
