import numpy as np
import time
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
agent1 = RandomAgent(player=1, random_seed=1)
agent2 = RandomAgent(player=2, random_seed=2)
n_iter = 1
n_samples = n_iter

t0 = time.time()
fields, masked_fields = mapgen.generate(n_samples)
for i in range(fields.shape[0]):
    field, masked_field = fields[i], masked_fields[i]
    game = SimpleGame(field, masked_field, tile_ids)
    while True:
        print(game.turn_count, game.positions[0], sorted(list(game.calc_player_actions(1))), np.sum(game.gold_field))
        if game.finished:
            break
        action1 = agent1.choose_action(game)
        print(action1)
        game.process_turn(1, action1)
        if game.finished:
            break
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
        print('#' * 15)
t1 = time.time()
total = t1-t0
print(total)

#print(game.masked_field)
#print(game.gold_field)
#print(game.first_gold, game.second_gold)
