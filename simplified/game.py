import numpy as np


class GameError(Exception):
    pass


class SimpleGame:
    """
    Class that encapsulates simplified Jackal game logic
    """
    def __init__(self,
                 full_field,
                 masked_field,
                 tile_ids,
                 max_turn=500):
        """
        :param full_field: np.array, (n+2) x (n+2) actual field that is hidden for players
        :param masked_field: np.array, (n+2) x (n+2) actual field with masks applied
        :param tile_ids: dict, tile_name - its id (must be non-negative)
        :param max_turn: int, maximal number of turns players allowed to make
        """
        self.full_field = full_field
        self.masked_field = masked_field
        self.tile_ids = tile_ids
        self.max_turn = max_turn

        self.tile_gold = {f'tile{i}': i for i in range(1, 6)}
        self.ids_tile = {self.tile_ids[key]: key for key in self.tile_ids}
        self.n = self.masked_field.shape[0] - 2

        self.gold_field = np.zeros((self.n + 2, self.n + 2))
        self.gold_left = self.calc_initial_gold()
        self.first_score, self.second_score = 0, 0
        self.game_finished = False

        first_positions = [((self.n+2)//2, 0)] * 4  # ship, pirate1, pirate2, pirate3
        second_positions = [((self.n+2)//2, self.n+1)] * 4
        self.positions = (first_positions, second_positions)

        first_dirs = {'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
                      'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)}
        second_dirs = {'N': (0, -1), 'NE': (-1, -1), 'E': (-1, 0), 'SE': (-1, 1),
                       'S': (0, 1), 'SW': (1, 1), 'W': (1, 0), 'NW': (1, -1)}
        self.dirs = (first_dirs, second_dirs)
        self.all_actions = self.calc_all_actions()
        self.first_actions = self.calc_player_actions(1)
        self.second_actions = self.calc_player_actions(2)
        self.turn_count = 0
        self.player_turn = 1

    def calc_initial_gold(self):
        """
        Count all initial gold

        :return: int, sum of all initial gold tiles
        """
        sum_ = 0
        for i in range(self.full_field.shape[0]):
            for j in range(self.full_field.shape[1]):
                tile = self.ids_tile[self.full_field[i, j]]
                sum_ += self.tile_gold.get(tile, 0)
        return sum_

    def calc_all_actions(self):
        """
        Prepare the listing of all possible actions

        :return: set, all actions in form of strings
        example: 'pirate1_NE_gold' -> pirate1 go to North-East with gold
        """
        actions = set()
        for i in range(1, 4):
            for dir_ in self.dirs[0]:
                actions.add(f'pirate{i}_{dir_}')
        for act in actions:
            actions.add(act+'_gold')
        return actions

    def calc_pos_directions(self, player, pos):
        """
        Prepare the listing of player's possible directions from position

        :param player: int, id of a player - 1 or 2
        :param pos: tuple(int, int), coordinates of position
        :return: set, available directions of a player from position in form of strings
        """
        positions = self.positions[player - 1]
        dirs = self.dirs[player-1]
        i, j = pos
        onship = pos == positions[0]
        directions = set()
        if onship:
            directions.add('N')
            for dir_ in {'W', 'E'}:
                i_delta, j_delta = dirs[dir_]
                pos_new = (i + i_delta, j + j_delta)
                if not self.tile_ship_prohibited(pos_new):
                    directions.add(dir_)
        else:
            for dir_ in dirs:
                i_delta, j_delta = dirs[dir_]
                pos_new = (i + i_delta, j + j_delta)
                if self.tile_in_sea(pos_new):
                    if pos_new == positions[0]:
                        directions.add(dir_)
                else:
                    directions.add(dir_)
        return directions

    def calc_player_actions(self, player):
        """
        Prepare the listing of player's possible actions

        :param player: int, id of a player - 1 or 2
        :return: set, available actions of a player in form of strings
        """
        positions = self.positions[player - 1]
        dirs = self.dirs[player - 1]
        actions = set()
        for pir in range(1, 4):
            pos = positions[pir]
            i, j = pos
            directions = self.calc_pos_directions(player, pos)
            for dir_ in directions:
                actions.add(f'pirate{pir}_{dir_}')
            if self.gold_field[pos] != 0:
                for dir_ in directions:
                    i_delta, j_delta = dirs[dir_]
                    pos_new = (i + i_delta, j + j_delta)
                    if not self.tile_in_mask(pos_new) and not self.tile_under_enemy(player, pos):
                        actions.add(f'pirate{pir}_{dir_}_gold')
        return actions

    def tile_in_sea(self, pos):
        """
        Check whether position is in sea

        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        i, j = pos
        if i == 0 or i == self.n + 1 or j == 0 or j == self.n + 1:
            return True
        if (i == 1 or i == self.n) and (j == 1 or j == self.n):
            return True
        return False

    def tile_ship_prohibited(self, pos):
        """
        Check whether position is prohibited for ships

        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        i, j = pos
        if j <= 1 or j >= self.n:
            return True
        return False

    def tile_in_mask(self, pos):
        """
        Check whether position is under mask

        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        if self.masked_field[pos] == 0:
            return True
        return False

    def tile_under_enemy(self, player, pos):
        """
        Check whether position is under enemy

        :param player: int, id of a player - 1 or 2
        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        enemy_positions = self.positions[player % 2 + 1]
        if pos in enemy_positions:
            return True
        return False
