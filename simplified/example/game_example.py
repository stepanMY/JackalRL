from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
n_samples = 1
fields, masked_fields = mapgen.generate(n_samples)
field, masked_field = fields[0], masked_fields[0, :, :]
print(field)
print(masked_field)

game = SimpleGame(field, masked_field, tile_ids)
player, action = 1, 'pir2_N'
game.process_turn(player, action)
player, action = 2, 'pir1_N'
game.process_turn(player, action)
print('# ' * 10)
print(game.masked_field)
