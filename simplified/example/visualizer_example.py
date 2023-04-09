from simplified.util.generator import MapGenerator
from simplified.util.visualizer import MapVisualizer

n = 5  # 11
tile_ids = {'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}  # 111
mapgen = MapGenerator(n, tile_counts, tile_ids)
n_samples = 1
fields, currfields = mapgen.generate(n_samples)
field, currfield = fields[0], currfields[0, :, :, 0]
print(field)
print('# # # # #')
print(currfield)

dir_ = 'C:\\Users\\stepanmy\\PycharmProjects\\JackalRL\\images\\'
tile_images = {0: 'unk', 1: 'sea', 2: 'ground', 3: 'gold1', 4: 'gold2', 5: 'gold3'}
mapviz = MapVisualizer(dir_, tile_images)
img_shown = mapviz.draw_field(field, (1000, 1400))
img_shown.show()
img_hidden = mapviz.draw_field(currfield, (1000, 1400))
img_hidden.show()
