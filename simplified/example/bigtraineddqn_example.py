import torch
from copy import deepcopy
import numpy as np
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.dqn import DqnEncoder, DqnAgent
from simplified.agents.baseline_agents import RandomAgent
from simplified.util.visualizer import MapVisualizer
from simplified.util.viewer import GameViewer

device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\images\\'
font_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\fonts\\'
weights_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\simplified\\network_weights\\'
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tile_images = {0: 'unk', 1: 'sea', 2: 'ground', 3: 'gold1', 4: 'gold2', 5: 'gold3', 6: 'gold4', 7: 'gold5'}
position_images = {(1, 0): 'boat_white', (1, 1): 'white1', (1, 2): 'white2', (1, 3): 'white3',
                   (2, 0): 'boat_black', (2, 1): 'black1', (2, 2): 'black2', (2, 3): 'black3'}
font_name = 'TimesNewRoman'

n = 11
tile_counts = {'ground': 101, 'gold1': 5, 'gold2': 5, 'gold3': 3, 'gold4': 2, 'gold5': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5, 'gold4': 6, 'gold5': 7}
n_games = 1
max_turn = 1000

mapgen = MapGenerator(n, tile_counts, tile_ids)
fields, masked_fields = mapgen.generate(n_games)
games = []
for i in range(n_games):
    field, masked_field = fields[i], masked_fields[i]
    games.append(SimpleGame(field, masked_field, tile_ids, max_turn))

dqn_encoder = DqnEncoder(games[0], n_games)
dqn_agent = DqnAgent(games[0], dqn_encoder)
dqn_agent.network.load_state_dict(torch.load(weights_dir + 'big_network5.pt', map_location=device))
agent2 = RandomAgent(player=2, random_seed=90)

game_progress = []
progress_tuple = (np.copy(games[0].masked_field), np.copy(games[0].full_field), np.copy(games[0].gold_field),
                  deepcopy(games[0].positions), games[0].turn_count, games[0].gold_left,
                  (games[0].first_gold, games[0].second_gold))
finished_games = set()
for j in range(games[0].max_turn // 2 + 1):
    gameids1, encoding1, possible_actions1 = dqn_encoder.encode(games, 1)
    with torch.no_grad():
        actionids1, actions1 = dqn_agent.choose_actions(encoding1, possible_actions1, greedy=True)
    gameid1_action = dict(zip(gameids1, actions1))
    for gameid in range(len(games)):
        if gameid in finished_games:
            continue
        game = games[gameid]
        action1 = gameid1_action[gameid]
        game.process_turn(1, action1)
        progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                          deepcopy(game.positions), game.turn_count, game.gold_left,
                          (game.first_gold, game.second_gold))
        game_progress.append(progress_tuple)
        if game.finished:
            finished_games.add(gameid)
    if len(finished_games) == dqn_encoder.n_games:
        break

    dqn_encoder.update_previous_turns(games)
    for gameid in range(len(games)):
        if gameid in finished_games:
            continue
        game = games[gameid]
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
        progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                          deepcopy(game.positions), game.turn_count, game.gold_left,
                          (game.first_gold, game.second_gold))
        game_progress.append(progress_tuple)
        if game.finished:
            finished_games.add(gameid)
    if len(finished_games) == dqn_encoder.n_games:
        break

map_viz = MapVisualizer(image_dir, tile_images, position_images, font_dir, font_name)
game_viewer = GameViewer(map_viz, game_progress)
game_viewer.start()
