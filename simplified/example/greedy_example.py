import numpy as np
import time
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import GreedyAgent
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
agent1 = GreedyAgent(player=1)
agent2 = GreedyAgent(player=2)
n_iter = 1024
n_samples = n_iter

t0 = time.time()
fields, masked_fields = mapgen.generate(n_samples)
for i in range(fields.shape[0]):
    field, masked_field = fields[i], masked_fields[i]
    game = SimpleGame(field, masked_field, tile_ids)
    while True:
        if game.finished:
            break
        action1 = agent1.choose_action(game)
        game.process_turn(1, action1)
        if game.finished:
            break
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
t1 = time.time()
total = t1-t0
print(total)
