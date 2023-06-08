import numpy as np


class MapGenerator:
    """
    Class that generates field for simplified Jackal Game
    """
    def __init__(self,
                 n,
                 tile_counts,
                 tile_ids):
        """
        :param n: int, size of square field
        :param tile_counts: dict, tile_name - number in the resulting field
        :param tile_ids: dict, tile_name - its id (must be non-negative)
        """
        self.n = n
        self.tile_counts = tile_counts
        counts_sum = 0
        for key in self.tile_counts:
            counts_sum += self.tile_counts[key]
        assert counts_sum == n * n - 4
        self.tile_ids = tile_ids
        self.tiles = []
        for tile_name in self.tile_counts:
            addition = [self.tile_ids[tile_name]] * self.tile_counts[tile_name]
            self.tiles.extend(addition)

    def create_tileorder(self, n_samples):
        """
        Create random order of tiles in each sample

        :param n_samples: int, number of orders to create
        :return: np.array, n_samples x (n * n - 4) tile order
        """
        indxs = np.random.rand(n_samples, len(self.tiles)).argsort(axis=1)
        tile_order = np.take(self.tiles, indxs)
        return tile_order

    def create_fullfields(self, n_samples, order):
        """
        Create n_samples fields of shape (n+2) x (n+2)

        :param n_samples: int, number of fields to create
        :param order: np.array, n_samples x (n * n - 4) order in which tiles should be placed in fields
        :return: np.array, n_samples x (n+2) x (n + 2) fields
        """
        fields = np.ones((n_samples, self.n+2, self.n+2))
        fields[:, 2:self.n, 1] = order[:, :self.n-2]
        fields[:, 2:self.n, self.n] = order[:, (self.n*self.n-4)-self.n+2:]
        fields[:, 1:self.n+1, 2:self.n] = order[:, self.n-2:(self.n*self.n-4)-self.n+2].reshape((n_samples,
                                                                                                 self.n,
                                                                                                 self.n-2))
        return fields

    def create_hiddenfields(self, fields):
        """
        Prepare fields for game with mask application

        :param fields: np.array, n_samples x (n + 2) x (n + 2) fields
        :return: np.array, n_samples x (n + 2) x (n + 2) fields with masks
        """
        fields_ = np.copy(fields)
        fields_[:, 2:self.n, 1] = 0
        fields_[:, 2:self.n, self.n] = 0
        fields_[:, 1:self.n+1, 2:self.n] = 0
        return fields_

    def generate(self, n_samples):
        """
        Generate n_samples fields

        :param n_samples: int, number of fields to create
        :return: (np.array, np.array), n_samples x (n + 2) x (n + 2) fields and n_samples x (n + 2) x (n + 2) x 2
        """
        order = self.create_tileorder(n_samples)
        fields = self.create_fullfields(n_samples, order)
        curr_fields = self.create_hiddenfields(fields)
        return fields, curr_fields
