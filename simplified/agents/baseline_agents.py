import numpy as np


class RandomAgent:
    """
    Class that encapsulate logic of agent that makes random possible actions
    """
    def __init__(self,
                 player,
                 random_seed):
        """
        :param player: int, id of a player - 1 or 2
        :param random_seed: int, seed that will be used for action sampling
        """
        self.player = player
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=self.random_seed)

    def choose_action(self, game):
        """
        Decide what action to make

        :param game: object of SimpleGame, current state of the game
        :return: string, action to make
        """
        possible_actions = sorted(list(game.calc_player_actions(self.player)))
        action = self.rng.choice(possible_actions)
        return action
