import numpy as np
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
n = 11
tile_counts = {'ground': 101, 'gold1': 5, 'gold2': 5, 'gold3': 3, 'gold4': 2, 'gold5': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5, 'gold4': 6, 'gold5': 7}
n_games = 256
max_turn = 1000

mapgen = MapGenerator(n, tile_counts, tile_ids)
fields, masked_fields = mapgen.generate(n_games)
games = []
for i in range(n_games):
    field, masked_field = fields[i], masked_fields[i]
    games.append(SimpleGame(field, masked_field, tile_ids, max_turn))
agent1 = RandomAgent(player=1, random_seed=1)
agent2 = RandomAgent(player=2, random_seed=2)
finished_games = set()

for j in range(max_turn // 2):
    if len(finished_games) == n_games:
        break
    indexes1, encoding1, actions1 = dqn_encoder.encode(games, 1)
    for i in range(len(games)):
        if i in finished_games:
            continue
        game = games[i]
        action1 = agent1.choose_action(game)
        game.process_turn(1, action1)
        if game.finished:
            finished_games.add(i)
    if len(finished_games) == n_games:
        break
    indexes2, encoding2, actions2 = dqn_encoder.encode(games, 2)
    for i in range(len(games)):
        if i in finished_games:
            continue
        game = games[i]
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
        if game.finished:
            finished_games.add(i)
