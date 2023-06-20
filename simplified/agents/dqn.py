import numpy as np
import torch
from ..util.nn import ResUnet


class EncoderError(Exception):
    pass


def inverse_encoding(encoding):
    """
    Inverse encoding for second player

    :param encoding: np.array, encoding of game states
    :return: np.array, inverted encoding
    """
    inverted_encoding = encoding.copy()
    inverted_encoding[:, 2:6, :, :] = encoding[:, 6:10, :, :]
    inverted_encoding[:, 6:10, :, :] = encoding[:, 2:6, :, :]
    inverted_encoding[:, 10, :, :] = encoding[:, 11, :, :]
    inverted_encoding[:, 11, :, :] = encoding[:, 10, :, :]
    inverted_encoding = inverted_encoding[:, :, ::-1, ::-1]
    return inverted_encoding


class DqnEncoder:
    """
    Class that encodes game progress for DQN-agent prediction:

    masked_field: n x n
    gold_field: n x n
    player_positions: 4 x n x n
    enemy_positions: 4 x n x n
    player_gold_share: n x n
    enemy_gold_share: n x n
    remained_gold_share: n x n
    onfield_gold_share: n x n
    undiscovered_gold_share: n x n
    game_length_share: n x n
    encoding_of_previous_k_turns: k x n x n
    total shape: n_games x (16 + k) x n x n
    """
    def __init__(self,
                 baseline_game,
                 n_games,
                 previous_k_turns=10):
        """
        :param baseline_game: SimpleGame object, example of the game that will be used to calculate constants
        :param n_games: int, number of games player simultaneously
        :param previous_k_turns: int, number of previous turns that will
        """
        self.baseline_game = baseline_game
        self.n_games = n_games
        self.previous_k_turns = previous_k_turns
        self.n = self.baseline_game.n + 2
        self.total_gold = self.baseline_game.initial_gold
        self.max_turn = self.baseline_game.max_turn
        self.unk_id = self.baseline_game.tile_ids['unk']
        self.previous_turns = {i: np.zeros((self.previous_k_turns, self.n, self.n), dtype=np.float32)
                               for i in range(self.n_games)}
        dirs = sorted(self.baseline_game.dirs[0].keys())
        all_actions = []
        for pir_id in range(1, 4):
            for dir_ in dirs:
                all_actions.append(f'{pir_id}_{dir_}')
            for dir_ in dirs:
                all_actions.append(f'{pir_id}_{dir_}_g')
        self.id_action = {i: all_actions[i] for i in range(len(all_actions))}
        self.action_id = {self.id_action[key]: key for key in self.id_action}

    def encode(self, games, player):
        """
        Create np.array of game encodings

        :param games: list of SimpleGame objects, games states that will be encoded
        :param player: int, id of a player - 1 or 2
        :return: (list, np.array, np.array), indexes of games, array encoding and encoding of possible actions
        """
        indexes = []
        for i in range(len(games)):
            game = games[i]
            if not game.finished:
                indexes.append(i)
        if len(indexes) == 0:
            raise EncoderError('All games have already finished')
        encoding = np.zeros((len(indexes), 16 + self.previous_k_turns, self.n, self.n), dtype=np.float32)
        available_actions = np.zeros((len(indexes), len(self.action_id)))
        encoding[:, 10:16, :, :] = 1
        player_gold, enemy_gold, remained_gold, onfield_gold, undiscovered_gold, game_length = [], [], [], [], [], []
        encoding_index = 0
        for i in indexes:
            game = games[i]
            encoding[encoding_index, 0, :, :] = game.masked_field
            encoding[encoding_index, 1, :, :] = game.gold_field
            for j in range(len(game.positions[0])):
                pos = game.positions[0][j]
                encoding[encoding_index, 2+j, pos[0], pos[1]] = 1
            for j in range(len(game.positions[1])):
                pos = game.positions[1][j]
                encoding[encoding_index, 6+j, pos[0], pos[1]] = 1
            player_gold.append(game.first_gold), enemy_gold.append(game.second_gold)
            onfield = np.sum(game.gold_field)
            remained_gold.append(game.gold_left), onfield_gold.append(onfield)
            undiscovered_gold.append((game.gold_left - onfield))
            game_length.append(game.turn_count)
            if game.turn_count == 0:
                encoding[encoding_index, 16:16 + self.previous_k_turns, :, :] = self.previous_turns[i]
            else:
                encoding[encoding_index, 16:16 + self.previous_k_turns - 1, :, :] \
                    = self.previous_turns[i][1:, :, :]
                pos, pos_new, gold_flag, attack_flag = game.last_turn
                encoding[encoding_index, 16 + self.previous_k_turns - 1, pos_new[0], pos_new[1]] = 1
                if attack_flag == 1:
                    encoding[encoding_index, 16 + self.previous_k_turns - 1, pos[0], pos[1]] = -2
                elif gold_flag == 1:
                    encoding[encoding_index, 16 + self.previous_k_turns - 1, pos[0], pos[1]] = -3
                else:
                    encoding[encoding_index, 16 + self.previous_k_turns - 1, pos[0], pos[1]] = -1
                self.previous_turns[i] = encoding[encoding_index, 16: 16 + self.previous_k_turns, :, :]
            possible_actions = game.calc_player_actions(player)
            for action in possible_actions:
                available_actions[encoding_index, self.action_id[action]] = 1
            encoding_index += 1
        mask = (encoding[:, 0, :, :] != self.unk_id).nonzero()
        encoding[mask[0], 0, mask[1], mask[2]] = 1
        encoding[:, 10, :, :] *= (np.array(player_gold)/self.total_gold)[:, None, None]
        encoding[:, 11, :, :] *= (np.array(enemy_gold)/self.total_gold)[:, None, None]
        encoding[:, 12, :, :] *= (np.array(remained_gold)/self.total_gold)[:, None, None]
        encoding[:, 13, :, :] *= (np.array(onfield_gold)/self.total_gold)[:, None, None]
        encoding[:, 14, :, :] *= (np.array(undiscovered_gold)/self.total_gold)[:, None, None]
        encoding[:, 15, :, :] *= (np.array(game_length)/self.max_turn)[:, None, None]
        if player == 2:
            encoding = inverse_encoding(encoding)
        return indexes, encoding, available_actions

    def update_previous_turns(self, games):
        """
        Update previous turns arrays. Will be used in test times

        :param games: list of SimpleGame objects, games states that will be used for updates
        :return: None
        """
        indexes = []
        for i in range(len(games)):
            game = games[i]
            if not game.finished:
                indexes.append(i)
        if len(indexes) == 0:
            raise EncoderError('All games have already finished')
        for i in indexes:
            game = games[i]
            previous_turns = np.zeros((self.previous_k_turns, self.n, self.n), dtype=np.float32)
            previous_turns[:-1, :, :] = self.previous_turns[i][1:, :, :]
            pos, pos_new, gold_flag, attack_flag = game.last_turn
            previous_turns[-1, pos_new[0], pos_new[1]] = 1
            if attack_flag == 1:
                previous_turns[-1, pos[0], pos[1]] = -2
            elif gold_flag == 1:
                previous_turns[-1, pos[0], pos[1]] = -3
            else:
                previous_turns[-1, pos[0], pos[1]] = -1
            self.previous_turns[i] = previous_turns
        return


class DqnAgent:
    """
    Class that encapsulates the logic of DQN-agent decision-making
    """
    def __init__(self,
                 baseline_game,
                 encoder,
                 device='cpu',
                 gamma=0.95,
                 eps=0.05):
        """
        :param baseline_game: SimpleGame object, example of the game that will be used to calculate constants
        :param encoder: DqnEncoder object, example of the encoder that will be used to calculate constants
        :param device, string, device to be used for inference and training 'cpu'/'gpu'
        :param gamma: float, parameter of optimization dynamics
        :param eps: float, parameter of epsilon-greedy strategy
        """
        self.baseline_game = baseline_game
        self.encoder = encoder
        self.device = device
        self.gamma = gamma
        self.eps = eps

        self.field_shape = (baseline_game.masked_field.shape[0], baseline_game.masked_field.shape[1])
        self.id_action = encoder.id_action
        self.previous_k_turns = encoder.previous_k_turns
        self.network = ResUnet(16 + self.previous_k_turns, len(self.id_action), self.field_shape).to(self.device)
        self.network.eval()
        self.target_network = ResUnet(16 + self.previous_k_turns, len(self.id_action), self.field_shape).to(self.device)
        self.target_network.eval()
        self.update_target()

    def choose_actions(self, encoding, possible_actions, greedy=False):
        """
        Choose action in every game based on q-values and epsilon-greedy strategy

        :param encoding: np.array, encoding of game states
        :param possible_actions: np.array, encoding of possible actions
        :param greedy: bool, whether to use greedy or eps-greedy mode
        :return: np.array, list(string), array of ids and list of actions to make in each of encoded games
        """
        qvalues = self.qvalues(encoding).detach().cpu().numpy()
        arry = np.array(qvalues)
        arry[possible_actions != 1] = -np.inf
        greedy_actionids = np.argmax(arry, axis=1)
        if greedy:
            actions = list(np.vectorize(self.id_action.get)(greedy_actionids))
        else:
            random_action_weights = np.random.rand(arry.shape[0], arry.shape[1])
            random_action_weights[possible_actions != 1] = -np.inf
            random_actionids = np.argmax(random_action_weights, axis=1)
            random_eps_weights = np.random.rand(random_actionids.shape[0])
            result_actionids = greedy_actionids
            result_actionids[random_eps_weights <= self.eps] = random_actionids[random_eps_weights <= self.eps]
            actions = list(np.vectorize(self.id_action.get)(greedy_actionids))
        return greedy_actionids, actions

    def qvalues(self, encoding):
        """
        Calculate q-values using usual network

        :param encoding: np.array, encoding of game states
        :return: torch.tensor, state-action q-values
        """
        encoding_ = encoding.copy()
        tensr = torch.from_numpy(encoding_).to(self.device)
        qvals = self.network(tensr)
        return qvals

    def target_qvalues(self, encoding):
        """
        Calculate q-values using target network

        :param encoding: np.array, encoding of game states
        :return: torch.tensor, state-action q-values
        """
        encoding_ = encoding.copy()
        tensr = torch.from_numpy(encoding_).to(self.device)
        qvals = self.target_network(tensr)
        return qvals

    def update_target(self):
        """
        Update target network weights

        :return: None
        """
        self.target_network.load_state_dict(self.network.state_dict())
