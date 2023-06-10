import numpy as np
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent
RANDOM_SEED = 500
np.random.seed(RANDOM_SEED)

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
n_samples = 1
fields, masked_fields = mapgen.generate(n_samples)
field, masked_field = fields[0], masked_fields[0, :, :]

game = SimpleGame(field, masked_field, tile_ids)
agent1 = RandomAgent(player=1, random_seed=1)
agent2 = RandomAgent(player=2, random_seed=2)
print(game.full_field)
print(game.masked_field)

while True:
    if game.finished:
        break
    action1 = agent1.choose_action(game)
    game.process_turn(1, action1)
    if game.finished:
        break
    action2 = agent2.choose_action(game)
    game.process_turn(2, action2)

print('# ' * 10)
print(game.turn_count)
print(game.full_field)
print(game.masked_field)
