import numpy as np
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent
RANDOM_SEED = 200
np.random.seed(RANDOM_SEED)

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
agent1 = RandomAgent(player=1, random_seed=70)
agent2 = RandomAgent(player=2, random_seed=90)
n_iter = 1
n_samples = n_iter

fields, masked_fields = mapgen.generate(n_samples)
for i in range(fields.shape[0]):
    field, masked_field = fields[i], masked_fields[i]
    game = SimpleGame(field, masked_field, tile_ids)
    while True:
        break
        if game.finished:
            break
        action1 = agent1.choose_action(game)
        game.process_turn(1, action1)
        if game.finished:
            break
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)

print(game.default_actions[0][(5, 4)])
