import numpy as np
from collections import deque


class AgentError(Exception):
    pass


def choose_direction(pos, pos_new, directions):
    """
    Choose direction to get from pos to next_pos

    :param pos: tuple(int, int), starting position
    :param pos_new: tuple(int, int), resulting position
    :param directions: dictionary, listing of possible directions
    :return: direction, string
    """
    i, j = pos
    i_new, j_new = pos_new
    for dir_ in directions:
        i_delta, j_delta = directions[dir_]
        if i + i_delta == i_new and j + j_delta == j_new:
            return dir_
    return


class RandomAgent:
    """
    Class that encapsulates logic of agent that makes random possible actions
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


class GreedyAgent:
    """
    Class that encapsulates logic of greedy agent:
    If it sees gold, tries to bring it to the ship.
    Otherwise, heads towards the nearest to ship unknown tile
    """
    def __init__(self, player):
        """
        :param player: int, id of a player - 1 or 2
        """
        self.player = player

    def choose_action(self, game):
        """
        Decide what action to make

        :param game: object of SimpleGame, current state of the game
        :return: string, action to make
        """
        dirs = game.dirs[self.player - 1]
        positions = game.positions[self.player - 1]
        gold_positions = []
        for pir_id in range(1, 4):
            if game.gold_field[positions[pir_id]] > 0:
                gold_positions.append(pir_id)
        pir_id, shortest_path = None, None
        if len(gold_positions) > 0:
            pir_id, shortest_path = self.find_shortest_path(game, 'ship', gold_positions, (0, 0))
        elif np.sum(game.gold_field) > 0:
            pir_id, shortest_path = self.find_shortest_path(game, 'gold', gold_positions, (0, 0))
        elif np.sum(game.masked_field == game.tile_ids['unk']) > 0:
            closest_unk = self.bfs(game, 0, 'unk', (0, 0))[-1]
            pir_id, shortest_path = self.find_shortest_path(game, 'exact', gold_positions, closest_unk)
        if pir_id is not None:
            pos, pos_new = shortest_path[0], shortest_path[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            if len(gold_positions) > 0:
                if game.tile_under_enemy(self.player, pos_new) or game.tile_in_mask(pos_new):
                    action = str(pir_id)+'_'+dir_
                    return action
                else:
                    action = str(pir_id)+'_'+dir_+'_g'
                    return action
            action = str(pir_id)+'_'+dir_
            return action
        raise AgentError('Unable to make a greedy move')

    def bfs(self, game, pir_id, mode, exact_pos):
        """
        Find the shortest paths to undiscovered field, gold field and ship

        :param game: object of SimpleGame, current state of the game
        :param pir_id: int, id of the pirate from whose position the shortest paths will be discovered
        :param mode: string, 'unk'/'exact'/'gold'/'ship'
        :param exact_pos: tuple(int, int), position to look for, used only in 'exact' mode
        :return: list of nodes, the shortest path
        """
        dirs = game.dirs[self.player - 1]
        positions = game.positions[self.player - 1]
        seen = {positions[pir_id]}
        previous = {positions[pir_id]: None}
        found_pos = None
        bfs_que = deque([positions[pir_id]])
        while len(bfs_que) != 0:
            pos = bfs_que.popleft()
            if mode == 'unk':
                if game.tile_in_mask(pos):
                    found_pos = pos
                    break
            elif mode == 'exact':
                if pos == exact_pos:
                    found_pos = pos
                    break
            elif mode == 'gold':
                if game.gold_field[pos] > 0:
                    found_pos = pos
                    break
            elif mode == 'ship':
                if pos == positions[0]:
                    found_pos = pos
                    break
            i, j = pos
            for dir_ in dirs:
                if pos == positions[0] and dir_ not in {'N', 'E', 'W'}:
                    continue
                i_delta, j_delta = dirs[dir_]
                pos_new = (i + i_delta, j + j_delta)
                if pos_new[0] < 0 or pos_new[1] < 0:
                    continue
                if pos_new[0] > game.masked_field.shape[0] - 1 or pos_new[1] > game.masked_field.shape[1] - 1:
                    continue
                if mode == 'ship' and game.tile_in_mask(pos_new):
                    continue
                if not game.tile_in_sea(pos_new) and pos_new not in seen:
                    seen.add(pos_new)
                    bfs_que.append(pos_new)
                    previous[pos_new] = pos
                elif game.tile_in_sea(pos_new) and pos_new == positions[0] and pos_new not in seen:
                    seen.add(pos_new)
                    bfs_que.append(pos_new)
                    previous[pos_new] = pos
                elif pos == positions[0] and not game.tile_ship_prohibited(pos_new) and pos_new not in seen:
                    seen.add(pos_new)
                    bfs_que.append(pos_new)
                    previous[pos_new] = pos
        shortest_path_ = [found_pos]
        while previous[shortest_path_[-1]] is not None:
            shortest_path_.append(previous[shortest_path_[-1]])
        shortest_path = shortest_path_[::-1]
        return shortest_path

    def find_shortest_path(self, game, mode, gold_positions, exact_pos):
        """
        Find the shortest path among all pirates

        :param game: object of SimpleGame, current state of the game
        :param mode: string, 'exact'/'gold'/'ship'
        :param gold_positions: list of tuples, ids of pirates with gold, used only in 'ship' mode
        :param exact_pos: tuple(int, int), position to look for, used only in 'exact' mode
        :return: pir_id, shortest_path (list of nodes)
        """
        shortest_paths = []
        if mode == 'ship':
            iterator = gold_positions
        else:
            iterator = range(1, 4)
        for pir_id in iterator:
            shortest_path = self.bfs(game, pir_id, mode, exact_pos)
            shortest_paths.append((pir_id, shortest_path))
        pir_id, shortest_path = min(shortest_paths, key=lambda x: len(x[1]))
        return pir_id, shortest_path


class SemiGreedyAgent(GreedyAgent):
    """
    Class that encapsulates logic of semi-greedy agent:
    If it has gold in one of his tiles, tries to bring it to the ship
    Otherwise it chooses between gold and unk tile based on tradeoff of costs
    """
    def __init__(self, player, gold_price=1.0, unk_price=1.5):
        """
        :param player: int, id of a player - 1 or 2
        :param gold_price: float, cost of movement towards gold
        :param unk_price: float, cost of movement towards unknown tile
        """
        super().__init__(player)
        self.gold_price = gold_price
        self.unk_price = unk_price

    def choose_action(self, game):
        """
        Choose action in semi-greedy manner

        :param game: object of SimpleGame, current state of the game
        :return:
        """
        dirs = game.dirs[self.player - 1]
        positions = game.positions[self.player - 1]
        gold_positions = []
        for pir_id in range(1, 4):
            if game.gold_field[positions[pir_id]] > 0:
                gold_positions.append(pir_id)
        if len(gold_positions) > 0:
            pir_id, shortest_path = self.find_shortest_path(game, 'ship', gold_positions, (0, 0))
            pos, pos_new = shortest_path[0], shortest_path[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            if game.tile_under_enemy(self.player, pos_new) or game.tile_in_mask(pos_new):
                action = str(pir_id)+'_'+dir_
                return action
            else:
                action = str(pir_id)+'_'+dir_+'_g'
                return action
        pir_id_gold, shortest_path_gold = None, None
        if np.sum(game.gold_field) > 0:
            pir_id_gold, shortest_path_gold = self.find_shortest_path(game, 'gold', gold_positions, (0, 0))
        pir_id_unk, shortest_path_unk = None, None
        if np.sum(game.masked_field == game.tile_ids['unk']) > 0:
            closest_unk = self.bfs(game, 0, 'unk', (0, 0))[-1]
            pir_id_unk, shortest_path_unk = self.find_shortest_path(game, 'exact', gold_positions, closest_unk)
        if pir_id_gold is not None and pir_id_unk is None:
            pos, pos_new = shortest_path_gold[0], shortest_path_gold[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            action = str(pir_id_gold)+'_'+dir_
            return action
        elif pir_id_unk is not None and pir_id_gold is None:
            pos, pos_new = shortest_path_unk[0], shortest_path_unk[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            action = str(pir_id_unk)+'_'+dir_
            return action
        elif pir_id_gold is not None and pir_id_unk is not None:
            gold_cost = self.gold_price*(len(shortest_path_gold) - 1)
            unk_cost = self.unk_price*(len(shortest_path_unk) - 1)
            if gold_cost <= unk_cost:
                pos, pos_new = shortest_path_gold[0], shortest_path_gold[1]
                dir_ = choose_direction(pos, pos_new, dirs)
                action = str(pir_id_gold) + '_' + dir_
                return action
            else:
                pos, pos_new = shortest_path_unk[0], shortest_path_unk[1]
                dir_ = choose_direction(pos, pos_new, dirs)
                action = str(pir_id_unk) + '_' + dir_
                return action
        raise AgentError('Unable to make a semi-greedy move')

