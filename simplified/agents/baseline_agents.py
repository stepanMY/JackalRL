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
        if len(gold_positions) > 0:
            shortest_paths = []
            for pir_id in gold_positions:
                shortest_path = self.bfs(game, pir_id, 'ship', (0, 0))
                shortest_paths.append((pir_id, shortest_path))
            pir_id, shortest_path = min(shortest_paths, key=lambda x: len(x[1]))
            pos, pos_new = shortest_path[0], shortest_path[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            if game.tile_under_enemy(self.player, pos_new) or game.tile_in_mask(pos_new):
                action = str(pir_id)+'_'+dir_
                return action
            else:
                action = str(pir_id)+'_'+dir_+'_g'
                return action
        if np.sum(game.gold_field) > 0:
            shortest_paths = []
            for pir_id in range(1, 4):
                shortest_path = self.bfs(game, pir_id, 'gold', (0, 0))
                shortest_paths.append((pir_id, shortest_path))
            pir_id, shortest_path = min(shortest_paths, key=lambda x: len(x[1]))
            pos, pos_new = shortest_path[0], shortest_path[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            action = str(pir_id)+'_'+dir_
            return action
        if np.sum(game.masked_field == game.tile_ids['unk']) > 0:
            appropriate = np.where(game.masked_field == game.tile_ids['unk'])
            if self.player == 1:
                indx = np.argmin(appropriate[1])
            else:
                indx = np.argmax(appropriate[1])
            exact_pos = (appropriate[0][indx], appropriate[1][indx])
            shortest_paths = []
            for pir_id in range(1, 4):
                shortest_path = self.bfs(game, pir_id, 'exact', exact_pos)
                shortest_paths.append((pir_id, shortest_path))
            pir_id, shortest_path = min(shortest_paths, key=lambda x: len(x[1]))
            pos, pos_new = shortest_path[0], shortest_path[1]
            dir_ = choose_direction(pos, pos_new, dirs)
            action = str(pir_id)+'_'+dir_
            return action
        raise AgentError('Unable to make a greedy move')

    def bfs(self, game, pir_id, mode, exact_pos):
        """
        Find the shortest paths to undiscovered field, gold field and ship

        :param game: object of SimpleGame, current state of the game
        :param pir_id: int, id of the pirate from whose position the shortest paths will be discovered
        :param mode: string, 'exact'/'gold'/'ship'
        :param exact_pos: tuple(int, int), position to look for, used only in 'exact' mode
        :return:
        """
        dirs = game.dirs[self.player - 1]
        positions = game.positions[self.player - 1]
        seen = {positions[pir_id]}
        previous = {positions[pir_id]: None}
        found_pos = None
        bfs_que = deque([positions[pir_id]])
        while len(bfs_que) != 0:
            pos = bfs_que.popleft()
            if mode == 'exact':
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
