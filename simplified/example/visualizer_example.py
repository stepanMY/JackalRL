from simplified.util.generator import MapGenerator
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

dir_ = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\images\\'
tile_images = {0: 'unk', 1: 'sea', 2: 'ground', 3: 'gold1', 4: 'gold2', 5: 'gold3'}
mapviz = MapVisualizer(dir_, tile_images)
img_shown = mapviz.draw_field(field, (1400, 1400))
img_shown.show()
img_hidden = mapviz.draw_field(masked_field, (1400, 1400))
img_hidden.show()
