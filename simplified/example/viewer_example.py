import numpy as np
from copy import deepcopy
from simplified.util.generator import MapGenerator
from simplified.util.visualizer import MapVisualizer
from simplified.util.viewer import GameViewer
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent

RANDOM_SEED = 15
np.random.seed(RANDOM_SEED)
n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
agent1 = RandomAgent(player=1, random_seed=3)
agent2 = RandomAgent(player=2, random_seed=4)
n_iter = 1
n_samples = n_iter

game_progress = []
fields, masked_fields = mapgen.generate(n_samples)
for i in range(fields.shape[0]):
    field, masked_field = fields[i], masked_fields[i]
    game = SimpleGame(field, masked_field, tile_ids)
    progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                      deepcopy(game.positions), game.turn_count, game.gold_left,
                      (game.first_gold, game.second_gold))
    game_progress.append(progress_tuple)
    while True:
        if game.finished:
            break
        action1 = agent1.choose_action(game)
        game.process_turn(1, action1)
        progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                          deepcopy(game.positions), game.turn_count, game.gold_left,
                          (game.first_gold, game.second_gold))
        game_progress.append(progress_tuple)
        if game.finished:
            break
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
        progress_tuple = (np.copy(game.masked_field), np.copy(game.full_field), np.copy(game.gold_field),
                          deepcopy(game.positions), game.turn_count, game.gold_left,
                          (game.first_gold, game.second_gold))
        game_progress.append(progress_tuple)

image_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\images\\'
tile_images = {0: 'unk', 1: 'sea', 2: 'ground', 3: 'gold1', 4: 'gold2', 5: 'gold3'}
position_images = {(1, 0): 'boat_white', (1, 1): 'white1', (1, 2): 'white2', (1, 3): 'white3',
                   (2, 0): 'boat_black', (2, 1): 'black1', (2, 2): 'black2', (2, 3): 'black3'}
font_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\fonts\\'
font_name = 'TimesNewRoman'
map_viz = MapVisualizer(image_dir, tile_images, position_images, font_dir, font_name)
game_viewer = GameViewer(map_viz, game_progress)
game_viewer.start()
