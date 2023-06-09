from simplified.util.generator import MapGenerator
from simplified.util.visualizer import MapVisualizer
from simplified.game import SimpleGame

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)
n_samples = 1
fields, maskedfields = mapgen.generate(n_samples)
field, maskedfield = fields[0], maskedfields[0, :, :]
print(field)
print(maskedfield)

game = SimpleGame(field, maskedfield, tile_ids)
player, action = 1, 'pir2_N'
game.process_turn(player, action)
player, action = 2, 'pir1_N'
game.process_turn(player, action)
print('# ' * 10)
print(game.masked_field)
