import torch
from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.agents.dqn import DqnEncoder, DqnAgent

n = 11
tile_counts = {'ground': 101, 'gold1': 5, 'gold2': 5, 'gold3': 3, 'gold4': 2, 'gold5': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5, 'gold4': 6, 'gold5': 7}
n_games = 16
max_turn = 1000

mapgen = MapGenerator(n, tile_counts, tile_ids)
fields, masked_fields = mapgen.generate(n_games)
games = []
for i in range(n_games):
    field, masked_field = fields[i], masked_fields[i]
    games.append(SimpleGame(field, masked_field, tile_ids, max_turn))
dqn_encoder = DqnEncoder(games[0], n_games)

agent = DqnAgent(games[0], dqn_encoder)

finished_games = set()

for _ in range(max_turn // 2):
    if len(finished_games) == n_games:
        break
    gameids1, encoding1, possible_actions1 = dqn_encoder.encode(games, 1)
    with torch.no_grad():
        actionids1, actions1 = agent.choose_actions(encoding1, possible_actions1)
    gameid1_action = dict(zip(gameids1, actions1))
    for gameid in range(len(games)):
        if gameid in finished_games:
            continue
        game = games[gameid]
        action1 = gameid1_action[gameid]
        game.process_turn(1, action1)
        if game.finished:
            finished_games.add(gameid)

    if len(finished_games) == n_games:
        break
    gameids2, encoding2, possible_actions2 = dqn_encoder.encode(games, 2)
    with torch.no_grad():
        actionids2, actions2 = agent.choose_actions(encoding2, possible_actions2)
    gameid2_action = dict(zip(gameids2, actions2))
    for gameid in range(len(games)):
        if gameid in finished_games:
            continue
        game = games[gameid]
        action2 = gameid2_action[gameid]
        game.process_turn(2, action2)
        if game.finished:
            finished_games.add(gameid)
