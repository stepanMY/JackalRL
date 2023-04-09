from PIL import Image


class MapVisualizer:
    """
    Class that generates images of game field
    """
    def __init__(self,
                 image_dir,
                 tile_images,
                 step=100):
        """
        :param image_dir: string, directory in which tile images are stored
        :param tile_images: dict, tile_name - its image name
        :param step: int, number of pixels in which tile
        """
        self.image_dir = image_dir
        self.tile_images = tile_images
        self.step = step
        self.tile_images_ = {key: Image.open(self.image_dir + self.tile_images[key] + '.png').resize((self.step,
                                                                                                      self.step))
                             for key in self.tile_images}

    def draw_field(self, field, size):
        """
        Returns image of the field

        :param field: np.array, n x (n+2) game field
        :param size: tuple(int, int), width and height of resulting image
        :return: PIL.Image.Image, resulting image of the field
        """
        new_im = Image.new('RGB', (field.shape[0]*self.step, field.shape[1]*self.step))
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                im = self.tile_images_[field[i][j]]
                i_im, j_im = i * self.step, (field.shape[1] - j - 1) * self.step
                new_im.paste(im, (i_im, j_im))
        new_im = new_im.resize(size)
        return new_im
