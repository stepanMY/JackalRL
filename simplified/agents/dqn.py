import numpy as np
import torch
import torch.nn as nn
import math
from collections import Counter
from ..util.nn import ResUnet
from ..game import SimpleGame


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
            return greedy_actionids, actions
        random_action_weights = np.random.rand(arry.shape[0], arry.shape[1])
        random_action_weights[possible_actions != 1] = -np.inf
        random_actionids = np.argmax(random_action_weights, axis=1)
        random_eps_weights = np.random.rand(random_actionids.shape[0])
        result_actionids = greedy_actionids
        result_actionids[random_eps_weights <= self.eps] = random_actionids[random_eps_weights <= self.eps]
        actions = list(np.vectorize(self.id_action.get)(result_actionids))
        return result_actionids, actions

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


class DqnTrainer:
    """
    Class that encapsulates logic of DQN training: self-play, TD-loss calculation and evaluation against baseline bot
    """
    def __init__(self,
                 dqn_agent,
                 encoder,
                 replay_buffer,
                 mapgen,
                 training_params,
                 criterion,
                 optimizer,
                 optimizer_params,
                 fit_procedure_params,
                 logger,
                 baseline_agent,
                 eps_params,
                 greedy_agent,
                 mode='selfplay',
                 max_grad_norm=50,
                 device='cpu'):
        """
        :param dqn_agent: DqnAgent, initialised dqn agent
        :param encoder: DqnEncoder, initialised dqn encoder
        :param replay_buffer: ReplayBuffer, initialised replay buffer
        :param mapgen: MapGenerator, initialised map generator
        :param training_params: dictionary, dictionary with keys - 'train_steps', 'train_samplesize'
        :param criterion: torch.loss, loss function to calculate and use in backwards
        :param optimizer, torch.optimizer, optimizer to use in gradient descent
        :param optimizer_params: dict, hyperparameters of optimizer
        :param fit_procedure_params: dict, dictionary with keys - 'max_games', 'train_after', 'eval_after',
                                                                  'targetupdate_after'
        :param logger: NeptuneLogger, initialised logger
        :param baseline_agent: RandomAgent/GreedyAgent/SemiGreedyAgent, initialised agent to be used in evaluation games
        :param eps_params: dict, dictionary with keys - 'eps_init', 'eps_final', 'eps_delta', 'eps_after'.
                                 overwrites default eps
        :param greedy_agent: GreedyAgent, will be used only if mode == 'greedyplay'
        :param mode: string, 'selfplay'/'greedyplay', way of learning
        :param max_grad_norm: int, maximum allowed gradient norm
        :param device: string, device to be used for inference and training 'cpu'/'gpu'
        """
        self.dqn_agent = dqn_agent
        self.encoder = encoder
        self.replay_buffer = replay_buffer
        self.mapgen = mapgen
        self.train_steps, self.train_samplesize = training_params['train_steps'], training_params['train_samplesize']
        self.criterion = criterion
        self.optimizer = optimizer(self.dqn_agent.network.parameters(), **optimizer_params)
        self.max_games = fit_procedure_params['max_games']
        self.train_after = fit_procedure_params['train_after']
        self.eval_after = fit_procedure_params['eval_after']
        self.targetupdate_after = fit_procedure_params['targetupdate_after']
        self.logger = logger
        self.baseline_agent = baseline_agent
        self.eps, self.eps_final, self.eps_delta, self.eps_after = eps_params['eps_init'], eps_params['eps_final'], \
            eps_params['eps_delta'], eps_params['eps_after']
        self.greedy_agent = greedy_agent
        self.mode = mode
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.games_counter = 0
        self.train_counter = self.train_after
        self.eval_counter = self.eval_after
        self.target_counter = self.targetupdate_after
        self.eps_counter = self.eps_after
        self.dqn_agent.eps = self.eps

    def selfplay_and_log(self, games):
        """
        Engage in dqn self-play and log every game

        :param games: list(SimpleGame), list of games to play
        :return: None
        """
        finished_games = set()
        first_transition, second_transition = None, None
        for j in range(games[0].max_turn // 2 + 1):
            gameids1, encoding1, possible_actions1 = self.encoder.encode(games, 1)
            if first_transition is None:
                first_transition = []
            else:
                first_transition.append(tuple([reward2_ids.copy(), -1 * np.array(reward2)]))
                first_transition.append(tuple([gameids1.copy(), encoding1.copy(), possible_actions1.copy()]))
                self.replay_buffer.add(tuple(first_transition))
                first_transition = []
            first_transition.append(tuple([gameids1.copy(), encoding1.copy(), possible_actions1.copy()]))
            with torch.no_grad():
                if j == 0:
                    qvalues = self.dqn_agent.qvalues(encoding1).detach().cpu().numpy()
                    arry = np.array(qvalues)
                    arry[possible_actions1 != 1] = -np.inf
                    self.logger.log_stepmetric('initial_state_V', np.mean(np.max(arry, axis=1)), self.games_counter)
                actionids1, actions1 = self.dqn_agent.choose_actions(encoding1, possible_actions1)
            first_transition.append(actionids1.copy())
            gameid1_action = dict(zip(gameids1, actions1))
            reward1_ids, reward1 = [], []
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action1 = gameid1_action[gameid]
                game.process_turn(1, action1)
                reward1_ids.append(gameid)
                reward1.append(game.last_reward)
                if game.finished:
                    finished_games.add(gameid)
            first_transition.append(np.array(reward1))
            if len(finished_games) == self.encoder.n_games:
                break

            gameids2, encoding2, possible_actions2 = self.encoder.encode(games, 2)
            if second_transition is None:
                second_transition = []
            else:
                second_transition.append(tuple([reward1_ids, -1 * np.array(reward1)]))
                second_transition.append(tuple([gameids2.copy(), encoding2.copy(), possible_actions2.copy()]))
                self.replay_buffer.add(tuple(second_transition))
                second_transition = []
            second_transition.append(tuple([gameids2.copy(), encoding2.copy(), possible_actions2.copy()]))
            with torch.no_grad():
                actionids2, actions2 = self.dqn_agent.choose_actions(encoding2, possible_actions2)
            second_transition.append(actionids2.copy())
            gameid2_action = dict(zip(gameids2, actions2))
            reward2_ids, reward2 = [], []
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action2 = gameid2_action[gameid]
                game.process_turn(2, action2)
                reward2_ids.append(gameid)
                reward2.append(game.last_reward)
                if game.finished:
                    finished_games.add(gameid)
            second_transition.append(np.array(reward2))
            if len(finished_games) == self.encoder.n_games:
                break
        self.games_counter += self.encoder.n_games
        self.logger.log_stepmetric('first_gold', np.mean([game.first_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('second_gold', np.mean([game.second_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('turn_count', np.mean([game.turn_count for game in games]), self.games_counter)
        self.logger.log_stepmetric('gold_left', np.mean([game.gold_left for game in games]), self.games_counter)
        self.logger.log_stepmetric('buffer_size', len(self.replay_buffer), self.games_counter)
        results = [game.result for game in games]
        results_counter = Counter(results)
        self.logger.log_stepmetric('first_wins', results_counter.get('first', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('second_wins', results_counter.get('second', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('draws', results_counter.get('draw', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('eps_current', self.dqn_agent.eps, self.games_counter)

    def greedyplay_and_log(self, games):
        """
        Engage in dqn play with greedy agent and log every game

        :param games: list(SimpleGame), list of games to play
        :return: None
        """
        finished_games = set()
        first_transition, second_transition = None, None
        for j in range(games[0].max_turn // 2 + 1):
            gameids1, encoding1, possible_actions1 = self.encoder.encode(games, 1)
            if first_transition is None:
                first_transition = []
            else:
                first_transition.append(tuple([reward2_ids.copy(), -1 * np.array(reward2)]))
                first_transition.append(tuple([gameids1.copy(), encoding1.copy(), possible_actions1.copy()]))
                self.replay_buffer.add(tuple(first_transition))
                first_transition = []
            first_transition.append(tuple([gameids1.copy(), encoding1.copy(), possible_actions1.copy()]))
            with torch.no_grad():
                if j == 0:
                    qvalues = self.dqn_agent.qvalues(encoding1).detach().cpu().numpy()
                    arry = np.array(qvalues)
                    arry[possible_actions1 != 1] = -np.inf
                    self.logger.log_stepmetric('initial_state_V', np.mean(np.max(arry, axis=1)), self.games_counter)
                actionids1, actions1 = self.dqn_agent.choose_actions(encoding1, possible_actions1)
            first_transition.append(actionids1.copy())
            gameid1_action = dict(zip(gameids1, actions1))
            reward1_ids, reward1 = [], []
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action1 = gameid1_action[gameid]
                game.process_turn(1, action1)
                reward1_ids.append(gameid)
                reward1.append(game.last_reward)
                if game.finished:
                    finished_games.add(gameid)
            first_transition.append(np.array(reward1))
            if len(finished_games) == self.encoder.n_games:
                break

            gameids2, encoding2, possible_actions2 = self.encoder.encode(games, 2)
            if second_transition is None:
                second_transition = []
            else:
                second_transition.append(tuple([reward1_ids, -1 * np.array(reward1)]))
                second_transition.append(tuple([gameids2.copy(), encoding2.copy(), possible_actions2.copy()]))
                self.replay_buffer.add(tuple(second_transition))
                second_transition = []
            second_transition.append(tuple([gameids2.copy(), encoding2.copy(), possible_actions2.copy()]))
            actionids2 = []
            reward2_ids, reward2 = [], []
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action2 = self.greedy_agent.choose_action(game)
                actionids2.append(self.encoder.action_id[action2])
                game.process_turn(2, action2)
                reward2_ids.append(gameid)
                reward2.append(game.last_reward)
                if game.finished:
                    finished_games.add(gameid)
            second_transition.append(np.array(actionids2))
            second_transition.append(np.array(reward2))
            if len(finished_games) == self.encoder.n_games:
                break
        self.games_counter += self.encoder.n_games
        self.logger.log_stepmetric('first_gold', np.mean([game.first_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('second_gold', np.mean([game.second_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('turn_count', np.mean([game.turn_count for game in games]), self.games_counter)
        self.logger.log_stepmetric('gold_left', np.mean([game.gold_left for game in games]), self.games_counter)
        self.logger.log_stepmetric('buffer_size', len(self.replay_buffer), self.games_counter)
        results = [game.result for game in games]
        results_counter = Counter(results)
        self.logger.log_stepmetric('first_wins', results_counter.get('first', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('second_wins', results_counter.get('second', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('draws', results_counter.get('draw', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('eps_current', self.dqn_agent.eps, self.games_counter)

    def train(self):
        """
        Loss calculation and network fit

        :return: None
        """
        losses, grad_norms = [], []
        for _ in range(self.train_steps):
            sample = self.replay_buffer.sample(self.train_samplesize)
            target, prediction = None, None
            for indx in range(len(sample)):
                current = sample[indx][0]
                actions = sample[indx][1]
                reward_before = sample[indx][2]
                reward_after_ids, reward_after = sample[indx][3]
                next_ = sample[indx][4]

                qvals = self.dqn_agent.qvalues(current[1])
                mask = torch.from_numpy(actions).to(self.device)[:, None]
                qvals = qvals.gather(1, mask)[:, 0]
                with torch.no_grad():
                    qvals_target = self.dqn_agent.target_qvalues(next_[1])
                    mask = torch.from_numpy(next_[2]).to(self.device)
                    qvals_target[mask != 1] = -math.inf
                    qvals_next, _ = torch.max(qvals_target, dim=1)
                if current[1].shape[0] == next_[1].shape[0]:
                    reward = torch.from_numpy(reward_before + reward_after).to(self.device)
                    y = reward + self.dqn_agent.gamma * qvals_next
                else:
                    next_rewardids, next_gameids = set(reward_after_ids), set(next_[0])
                    good_reward_indxs, good_game_indxs = [], []
                    for i in range(len(current[0])):
                        if current[0][i] in next_rewardids:
                            good_reward_indxs.append(i)
                        if current[0][i] in next_gameids:
                            good_game_indxs.append(i)
                    reward_after_ = np.zeros(reward_before.shape[0])
                    reward_after_[good_reward_indxs] = reward_after
                    reward = torch.from_numpy(reward_before + reward_after_).to(self.device)
                    qvals_next_ = torch.zeros(qvals.shape[0]).to(self.device)
                    qvals_next_[good_game_indxs] = qvals_next
                    y = reward + self.dqn_agent.gamma * qvals_next_
                if target is None:
                    target = y
                else:
                    target = torch.cat([target, y])
                if prediction is None:
                    prediction = qvals
                else:
                    prediction = torch.cat([prediction, qvals])
            loss = self.criterion(prediction.float(), target.float())
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.dqn_agent.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(loss.data.cpu().item())
            grad_norms.append(grad_norm.cpu())
        self.logger.log_stepmetric('loss', np.mean(losses), self.games_counter)
        self.logger.log_stepmetric('grad_norm', np.mean(grad_norms), self.games_counter)

    def eval(self, games):
        """
        Evaluate agent against baseline bot

        :param games: list(SimpleGame), list of games to play
        :return: None
        """
        finished_games = set()
        for j in range(games[0].max_turn // 2 + 1):
            gameids1, encoding1, possible_actions1 = self.encoder.encode(games, 1)
            with torch.no_grad():
                actionids1, actions1 = self.dqn_agent.choose_actions(encoding1, possible_actions1, greedy=True)
            gameid1_action = dict(zip(gameids1, actions1))
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action1 = gameid1_action[gameid]
                game.process_turn(1, action1)
                if game.finished:
                    finished_games.add(gameid)
            if len(finished_games) == self.encoder.n_games:
                break

            self.encoder.update_previous_turns(games)
            for gameid in range(len(games)):
                if gameid in finished_games:
                    continue
                game = games[gameid]
                action2 = self.baseline_agent.choose_action(game)
                game.process_turn(2, action2)
                if game.finished:
                    finished_games.add(gameid)
            if len(finished_games) == self.encoder.n_games:
                break

        self.logger.log_stepmetric('eval_first_gold', np.mean([game.first_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('eval_second_gold',
                                   np.mean([game.second_gold for game in games]), self.games_counter)
        self.logger.log_stepmetric('eval_turn_count', np.mean([game.turn_count for game in games]), self.games_counter)
        self.logger.log_stepmetric('eval_gold_left', np.mean([game.gold_left for game in games]), self.games_counter)
        results = [game.result for game in games]
        results_counter = Counter(results)
        self.logger.log_stepmetric('eval_first_wins', results_counter.get('first', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('eval_second_wins', results_counter.get('second', 0)/len(games), self.games_counter)
        self.logger.log_stepmetric('eval_draws', results_counter.get('draw', 0)/len(games), self.games_counter)

    def prepare_games(self):
        """
        Prepare list of games

        :return: list(SimpleGame), list of games to play
        """
        fields, masked_fields = self.mapgen.generate(self.encoder.n_games)
        games = []
        for i in range(self.encoder.n_games):
            field, masked_field = fields[i], masked_fields[i]
            games.append(SimpleGame(field, masked_field, self.mapgen.tile_ids, self.encoder.max_turn))
        return games

    def fit(self):
        """
        Self-play and train network

        :return: None
        """
        while self.games_counter < self.max_games:
            games = self.prepare_games()
            if self.mode == 'selfplay':
                self.selfplay_and_log(games)
            elif self.mode == 'greedyplay':
                self.greedyplay_and_log(games)
            if self.games_counter >= self.train_counter:
                self.train()
                self.train_counter += self.train_after
            if self.games_counter >= self.eval_counter:
                games = self.prepare_games()
                self.eval(games)
                self.eval_counter += self.eval_after
            if self.games_counter >= self.target_counter:
                self.dqn_agent.update_target()
                self.target_counter += self.targetupdate_after
                self.logger.log_stepmetric('target_update', 1, self.games_counter)
            if self.games_counter >= self.eps_counter:
                self.eps = max(self.eps_final, self.eps - self.eps_delta)
                self.dqn_agent.eps = self.eps
                self.eps_counter += self.eps_after
