import numpy as np
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.baseline_agents import RandomAgent
from simplified.util.logger import prepare_game_tuple
from simplified.agents.dqn import DqnEncoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
n_games = 128
max_turn = 1000

mapgen = MapGenerator(n, tile_counts, tile_ids)
fields, masked_fields = mapgen.generate(n_games)
games = []
for i in range(n_games):
    field, masked_field = fields[i], masked_fields[i]
    games.append(SimpleGame(field, masked_field, tile_ids, max_turn))
dqn_encoder = DqnEncoder(games[0], n_games)
agent1 = RandomAgent(player=1, random_seed=1)
agent2 = RandomAgent(player=2, random_seed=2)
limit = 5
actions = []

for j in range(max_turn // 2):
    indexes1, encoding1, actions1 = dqn_encoder.encode(games, 1)
    for i in range(len(games)):
        game = games[i]
        if game.finished:
            continue
        action1 = agent1.choose_action(game)
        game.process_turn(1, action1)
        if i == 0:
            actions.append((game.last_turn, action1))
    indexes2, encoding2, actions2 = dqn_encoder.encode(games, 2)
    #dqn_encoder.update_previous_turns(games)
    for i in range(len(games)):
        game = games[i]
        if game.finished:
            continue
        action2 = agent2.choose_action(game)
        game.process_turn(2, action2)
        if i == 0:
            actions.append((game.last_turn, action2))
    if j >= limit:
        break

indexes1, encoding1, actions1 = dqn_encoder.encode(games, 1)
indexes2, encoding2, actions2 = dqn_encoder.encode(games, 2)
print(actions1[0])
print(dqn_encoder.id_action)
print(dqn_encoder.action_id)
#print(games[0].masked_field)
#print(games[0].gold_field)
#print(games[0].first_gold, games[0].second_gold, games[0].gold_left, games[0].positions)
#print('#' * 10)
#print(encoding2[0, 6, :, :])
#print(encoding2[0, 7, :, :])
#print(encoding2[0, 8, :, :])
#print(encoding2[0, 9, :, :])
#print(encoding1.shape)
#print(encoding2.shape)

#print(actions)
#print('#' * 10)
#for i in range(10):
#    print(encoding2[0, 16 + i, :, :])
