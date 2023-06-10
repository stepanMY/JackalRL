from simplified.util.generator import MapGenerator
from simplified.game import SimpleGame
from simplified.util.visualizer import MapVisualizer

n = 5  # 11
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}  # 111
mapgen = MapGenerator(n, tile_counts, tile_ids)
n_samples = 1
fields, masked_fields = mapgen.generate(n_samples)
field, masked_field = fields[0], masked_fields[0, :, :]
print(field)
print('# # # # #')
print(masked_field)

image_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\images\\'
tile_images = {0: 'unk', 1: 'sea', 2: 'ground', 3: 'gold1', 4: 'gold2', 5: 'gold3'}
position_images = {(1, 0): 'boat_white', (1, 1): 'white1', (1, 2): 'white2', (1, 3): 'white3',
                   (2, 0): 'boat_black', (2, 1): 'black1', (2, 2): 'black2', (2, 3): 'black3'}
font_dir = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\fonts\\'
font_name = 'TimesNewRoman'
game = SimpleGame(field, masked_field, tile_ids)
mapviz = MapVisualizer(image_dir, tile_images, position_images, font_dir, font_name)
gold_field = game.gold_field
gold_field[5, 3] = 2
gold_field[3, 3] = 1
gold_field[2, 5] = 5
turn_count = 561
gold_left = 7
gold_counts = 2, 0
img_shown = mapviz.draw_field(field, gold_field, game.positions, turn_count, gold_left, gold_counts, (1400, 1400))
img_shown.show()
img_hidden = mapviz.draw_field(masked_field, gold_field, game.positions, turn_count, gold_left,
                               gold_counts, (1400, 1400))
img_hidden.show()
