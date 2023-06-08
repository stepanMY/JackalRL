from simplified.util.generator import MapGenerator

n = 5
tile_counts = {'ground': 15, 'gold1': 3, 'gold2': 2, 'gold3': 1}
tile_ids = {'unk': 0, 'sea': 1, 'ground': 2, 'gold1': 3, 'gold2': 4, 'gold3': 5}
mapgen = MapGenerator(n, tile_counts, tile_ids)

n_samples = 1
fields, maskedfields = mapgen.generate(n_samples)
print(fields.shape, maskedfields.shape)
print('# # # # #')
print(fields[0])
print('# # # # #')
print(maskedfields[0, :, :])
