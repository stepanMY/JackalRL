import numpy as np


class GameError(Exception):
    pass


def parse_action(action):
    """
    Translates string representation of action to game logic

    :param action: string, action to parse
    :return: id of pirate, direction, flag of gold movement
    """
    action_sp = action.split('_')
    pir_id, direction = int(action_sp[0][-1]), action_sp[1]
    if len(action_sp) == 3 and action_sp[-1] == 'g':
        gold_flag = True
    else:
        gold_flag = False
    return pir_id, direction, gold_flag


def parse_dir(action):
    """
    Parses direction from action string

    :param action: string, action to parse
    :return: direction
    """
    action_ = action[2:4]
    if action_[-1] == '_':
        return action_[:-1]
    return action_


class SimpleGame:
    """
    Class that encapsulates simplified Jackal game logic
    """
    def __init__(self,
                 full_field,
                 masked_field,
                 tile_ids,
                 max_turn=1000):
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

        self.tile_gold = {f'gold{i}': i for i in range(1, 6)}
        self.ids_tile = {self.tile_ids[key]: key for key in self.tile_ids}
        self.n = self.masked_field.shape[0] - 2

        self.gold_field = np.zeros((self.n + 2, self.n + 2))
        self.initial_gold = self.calc_initial_gold()
        self.gold_left = self.initial_gold
        self.first_gold, self.second_gold = 0, 0
        self.finished = False
        self.result = None

        first_positions = [((self.n+2)//2, 0)] * 4  # ship, pirate1, pirate2, pirate3
        second_positions = [((self.n+2)//2, self.n+1)] * 4
        self.positions = (first_positions, second_positions)

        first_dirs = {'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
                      'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)}
        second_dirs = {'N': (0, -1), 'NE': (-1, -1), 'E': (-1, 0), 'SE': (-1, 1),
                       'S': (0, 1), 'SW': (1, 1), 'W': (1, 0), 'NW': (1, -1)}
        self.dirs = (first_dirs, second_dirs)
        first_default_actions, second_default_actions = dict(), dict()
        self.default_actions = (first_default_actions, second_default_actions)
        self.update_default_actions(1), self.update_default_actions(2)
        first_actions, second_actions = {1: set(), 2: set(), 3: set()}, {1: set(), 2: set(), 3: set()}
        self.actions = (first_actions, second_actions)
        first_update_set, second_update_set = {1, 2, 3}, {1, 2, 3}
        self.update_sets = (first_update_set, second_update_set)
        self.update_player_possible_actions(1), self.update_player_possible_actions(2)
        self.turn_count = 0
        self.player_turn = 1

        self.last_turn = []  # pos, next_pos, gold_flag, attack_flag (for encoding)
        self.last_reward = 0  # reward achieved in last episode (for encoding)

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

    def update_default_actions(self, player):
        """
        Update default actions that player can make from certain positions

        :param player: int, id of a player - 1 or 2
        :return: None
        """
        dirs = self.dirs[player - 1]
        player_default_actions = self.default_actions[player - 1]
        for i in range(self.full_field.shape[0]):
            for j in range(self.full_field.shape[1]):
                pos = (i, j)
                player_default_actions[pos] = {k: {'always': set(),
                                                   'sometimes': {'ship': set(), 'gold': set(), 'ship_gold': set()}}
                                               for k in range(1, 4)}
                if self.tile_in_sea(pos) and self.tile_ship_prohibited(pos):
                    continue
                elif self.tile_in_sea(pos):
                    for pir_id in range(1, 4):
                        pir_id_ = str(pir_id)
                        player_default_actions[pos][pir_id]['always'].add(pir_id_+'_'+'N')
                    for dir_ in {'W', 'E'}:
                        i_delta, j_delta = dirs[dir_]
                        pos_new = (i + i_delta, j + j_delta)
                        if not self.tile_ship_prohibited(pos_new):
                            for pir_id in range(1, 4):
                                pir_id_ = str(pir_id)
                                player_default_actions[pos][pir_id]['always'].add(pir_id_+'_'+dir_)
                else:
                    for dir_ in dirs:
                        i_delta, j_delta = dirs[dir_]
                        pos_new = (i + i_delta, j + j_delta)
                        if self.tile_in_sea(pos_new) and not self.tile_ship_prohibited(pos_new) \
                                and not self.tile_under_enemy(player, pos_new):
                            for pir_id in range(1, 4):
                                pir_id_ = str(pir_id)
                                player_default_actions[pos][pir_id]['sometimes']['ship'].add(pir_id_+'_'+dir_)
                                player_default_actions[pos][pir_id]['sometimes']['ship_gold'].add(pir_id_+'_'+dir_+'_g')
                        elif not self.tile_in_sea(pos_new):
                            for pir_id in range(1, 4):
                                pir_id_ = str(pir_id)
                                player_default_actions[pos][pir_id]['always'].add(pir_id_+'_'+dir_)
                                player_default_actions[pos][pir_id]['sometimes']['gold'].add(pir_id_+'_'+dir_+'_g')
        return

    def update_player_possible_actions(self, player):
        """
        Update possible actions of the certain player

        :param player: int, id of a player - 1 or 2
        :return: None
        """
        update_set = self.update_sets[player - 1]
        for pir in update_set:
            self.update_possible_actions(player, pir)
        update_set.clear()
        return

    def update_possible_actions(self, player, pir_id):
        """
        Update possible actions of the certain pirate of the certain player

        :param player: int, id of a player - 1 or 2
        :param pir_id: int, id of the pirate whose possible actions need to be updated
        :return: None
        """
        actions = self.actions[player - 1]
        positions = self.positions[player - 1]
        pos = positions[pir_id]
        actions[pir_id] = self.calc_pos_actions(player, pir_id, pos)
        return

    def calc_pos_actions(self, player, pir_id, pos):
        """
        Prepare the listing of player's possible actions from position

        :param player: int, id of a player - 1 or 2
        :param pir_id: int, id of the pirate whose possible actions need to be calculated
        :param pos: tuple(int, int), coordinates of position
        :return: set, available actions of a player from position in form of strings
        """
        default_actions = self.default_actions[player - 1]
        positions = self.positions[player - 1]
        dirs = self.dirs[player-1]
        i, j = pos
        always = default_actions[pos][pir_id]['always']
        sometimes_ship = default_actions[pos][pir_id]['sometimes']['ship']
        sometimes_gold = default_actions[pos][pir_id]['sometimes']['gold']
        gold_copied_flag = False
        pir_id_ = str(pir_id)
        actions = set()
        if len(sometimes_ship) > 0:
            for dir_ in {'S', 'SE', 'SW'}:
                i_delta, j_delta = dirs[dir_]
                pos_new = (i + i_delta, j + j_delta)
                if pos_new == positions[0]:
                    actions.add(pir_id_+'_'+dir_)
                    if self.gold_field[pos] > 0:
                        actions.add(pir_id_+'_'+dir_+'_g')
                    break
        if self.gold_field[pos] > 0:
            for action in always:
                dir_ = parse_dir(action)
                i_delta, j_delta = dirs[dir_]
                pos_new = (i + i_delta, j + j_delta)
                if self.tile_in_mask(pos_new) or self.tile_under_enemy(player, pos_new):
                    if not gold_copied_flag:
                        sometimes_gold = sometimes_gold.copy()
                        gold_copied_flag = True
                    sometimes_gold.remove(pir_id_+'_'+dir_+'_g')
            actions = set.union(actions, always, sometimes_gold)
        else:
            actions = set.union(actions, always)
        return actions

    def calc_player_actions(self, player):
        """
        Prepare the listing of player's possible actions

        :param player: int, id of a player - 1 or 2
        :return: set, available actions of a player in form of strings
        """
        actions = self.actions[player - 1]
        possible_actions = set.union(actions[1], actions[2], actions[3])
        return possible_actions

    def process_turn(self, player, action):
        """
        Make turn in the game

        :param player: int, id of a player - 1 or 2
        :param action: string, action to make
        example: '1_NE_g' -> pirate1 go to North-East with gold
        :return: None
        """
        if self.finished:
            raise GameError('Game has already finished')
        if player != self.player_turn:
            raise GameError('Wrong player')
        possible_actions = self.calc_player_actions(player)
        if action not in possible_actions:
            raise GameError('Wrong action')
        positions = self.positions[player - 1]
        dirs = self.dirs[player - 1]
        self.last_turn = [0, 0, 0, 0]
        self.last_reward = 0
        pir_id, direction, gold_flag = parse_action(action)
        curr_pos = positions[pir_id]
        delta = dirs[direction]
        new_pos = (curr_pos[0] + delta[0], curr_pos[1] + delta[1])
        self.move_pir(player, pir_id, new_pos)
        self.last_turn[:2] = [curr_pos, new_pos]
        self.discover_tile(player, new_pos)
        if gold_flag:
            self.move_gold(player, curr_pos, new_pos)
            self.last_turn[2] = 1
        self.make_attack(player, new_pos)
        self.check_endgame_condition()
        self.update_player_possible_actions(player % 2 + 1)
        self.pass_turn()
        return

    def move_pir(self, player, pir_id, new_pos):
        """
        Move pirate

        :param player: int, id of a player - 1 or 2
        :param pir_id: int, id of the pirate that will be moved
        :param new_pos: tuple, coordinates of new position
        :return: None
        """
        player_update_set, enemy_update_set = self.update_sets[player - 1], self.update_sets[player % 2]
        dirs = self.dirs[player - 1]
        positions, enemy_positions = self.positions[player - 1], self.positions[player % 2]
        ship_position = positions[0]
        if positions[pir_id] == ship_position:
            if ship_position[0] != new_pos[0]:
                affected_positions = set.union({(positions[pir_id][0]+dirs[dir_][0], positions[pir_id][1]+dirs[dir_][1])
                                                for dir_ in ['N', 'NE', 'NW']},
                                               {(new_pos[0]+dirs[dir_][0], new_pos[1]+dirs[dir_][1])
                                                for dir_ in ['N', 'NE', 'NW']})
                for i in range(len(positions)):
                    if positions[i] == ship_position:
                        positions[i] = new_pos
                        if i > 0:
                            player_update_set.add(i)
                    elif positions[i] in affected_positions:
                        if i > 0:
                            player_update_set.add(i)
            else:
                positions[pir_id] = new_pos
                player_update_set.add(pir_id)
                for enemy_id in range(1, 4):
                    if self.gold_field[enemy_positions[enemy_id]] > 0:
                        enemy_update_set.add(enemy_id)
        else:
            positions[pir_id] = new_pos
            player_update_set.add(pir_id)
            for enemy_id in range(1, 4):
                if self.gold_field[enemy_positions[enemy_id]] > 0:
                    enemy_update_set.add(enemy_id)
        return

    def discover_tile(self, player, new_pos):
        """
        Discover tile if necessary

        :param player: int, id of a player - 1 or 2
        :param new_pos: tuple, coordinates of new position
        :return: None
        """
        player_update_set = self.update_sets[player - 1]
        positions = self.positions[player - 1]
        if self.masked_field[new_pos] == self.tile_ids['unk']:
            self.masked_field[new_pos] = self.full_field[new_pos]
            tile = self.ids_tile[self.full_field[new_pos]]
            self.gold_field[new_pos] = self.tile_gold.get(tile, 0)
            for pir_id in range(1, 4):
                if self.gold_field[positions[pir_id]] > 0:
                    player_update_set.add(pir_id)
        return

    def move_gold(self, player, curr_pos, new_pos):
        """
        Move gold

        :param player: int, id of a player - 1 or 2
        :param curr_pos: tuple, coordinates of previous position
        :param new_pos: tuple, coordinates of new position
        :return: None
        """
        player_update_set = self.update_sets[player - 1]
        positions = self.positions[player - 1]
        self.gold_field[curr_pos] -= 1
        if self.gold_field[curr_pos] == 0:
            for pir_id in range(1, 4):
                if positions[pir_id] == curr_pos:
                    player_update_set.add(pir_id)
        if new_pos == positions[0]:
            if player == 1:
                self.first_gold += 1
                self.last_reward = 1
            else:
                self.second_gold += 1
                self.last_reward = 1
            self.gold_left -= 1
        else:
            self.gold_field[new_pos] += 1
            if self.gold_field[new_pos] == 1:
                for pir_id in range(1, 4):
                    if positions[pir_id] == new_pos:
                        player_update_set.add(pir_id)
        return

    def make_attack(self, player, new_pos):
        """
        Attack opponent if necessary

        :param player: int, id of a player - 1 or 2
        :param new_pos: tuple, coordinates of new position
        :return: None
        """
        enemy_update_set = self.update_sets[player % 2]
        enemy_positions = self.positions[player % 2]
        for enemy_id in range(1, 4):
            if enemy_positions[enemy_id] == new_pos:
                enemy_positions[enemy_id] = enemy_positions[0]
                enemy_update_set.add(enemy_id)
                self.last_turn[3] = 1
        return

    def check_endgame_condition(self):
        """
        Check endgame condition

        :return: None
        """
        if self.turn_count >= self.max_turn or self.first_gold + self.second_gold == self.initial_gold:
            self.finished = True
            if self.first_gold == self.second_gold:
                self.result = 'draw'
            elif self.first_gold > self.second_gold:
                self.result = 'first'
            else:
                self.result = 'second'
        return

    def pass_turn(self):
        """
        Pass turn to other player

        :return: None
        """
        self.turn_count += 1
        self.player_turn = self.player_turn % 2 + 1
        return

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
        if i <= 1 or i >= self.n:
            return True
        return False

    def tile_in_mask(self, pos):
        """
        Check whether position is under mask

        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        if self.masked_field[pos] == self.tile_ids['unk']:
            return True
        return False

    def tile_under_enemy(self, player, pos):
        """
        Check whether position is under enemy

        :param player: int, id of a player - 1 or 2
        :param pos: tuple(int, int), coordinates of position
        :return: bool
        """
        enemy_positions = self.positions[player % 2]
        if pos in enemy_positions:
            return True
        return False
