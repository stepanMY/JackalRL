from PIL import Image, ImageDraw, ImageFont


class MapVisualizer:
    """
    Class that generates images of game field
    """
    def __init__(self,
                 image_dir,
                 tile_images,
                 position_images,
                 font_dir,
                 font_name,
                 gold_font_size=40,
                 info_font_size=20,
                 step=100):
        """
        :param image_dir: string, directory in which tile images are stored
        :param tile_images: dict, tile_name - its image name,
        :param position_images: dict, position_name - its image name
        :param font_dir: string, directory in which font is stored
        :param font_name: string, name of the font
        :param gold_font_size: int, size of the font of gold numbers
        :param info_font_size: int, size of the font of info
        :param step: int, number of pixels in which tile
        """
        self.image_dir = image_dir
        self.tile_images = tile_images
        self.position_images = position_images
        self.font_dir = font_dir
        self.font_name = font_name
        self.gold_font_size = gold_font_size
        self.info_font_size = info_font_size
        self.step = step
        self.pir_size = int(self.step/3)
        self.tile_images_ = {key: Image.open(self.image_dir + self.tile_images[key] + '.png').convert("RGBA")
                                                                                             .resize((self.step,
                                                                                                      self.step))
                             for key in self.tile_images}
        ship_images = {key: Image.open(self.image_dir + self.position_images[key] + '.png').convert("RGBA")
                                                                                           .resize((self.step,
                                                                                                    self.step))
                       for key in self.position_images if key[1] == 0}
        pir_images = {key: Image.open(self.image_dir + self.position_images[key] + '.png').convert("RGBA")
                                                                                          .resize((self.pir_size,
                                                                                                   self.pir_size))
                      for key in self.position_images if key[1] != 0}
        self.position_images_ = {**ship_images, **pir_images}
        self.gold_font = ImageFont.truetype(f'{self.font_dir}/{self.font_name}.ttf', self.gold_font_size)
        self.info_font = ImageFont.truetype(f'{self.font_dir}/{self.font_name}.ttf', self.info_font_size)
        self.rgb_orange = (255, 69, 0)
        self.rgb_red = (128, 0, 0)

    def draw_field(self, field, gold_field, positions, turn_count, gold_left, gold_counts, size):
        """
        Draws image of the field

        :param field: np.array, (n+2) x (n+2) game field
        :param gold_field: np.array, (n+2) x (n+2) field of gold counts
        :param positions: tuple(list, list), positions of ships and pirates
        :param turn_count: int, number of turns passed
        :param gold_left: int, gold left on the field
        :param gold_counts: tuple(int, int), gold counts of each player
        :param size: tuple(int, int), width and height of resulting image
        :return: PIL.Image.Image, resulting image of the field
        """
        new_im = Image.new('RGBA', (field.shape[0]*self.step, field.shape[1]*self.step))
        draw = ImageDraw.Draw(new_im)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                im = self.tile_images_[field[i][j]]
                i_im, j_im = i * self.step, (field.shape[1] - j - 1) * self.step
                if positions[0][0] == (i, j):
                    ship_im = self.position_images_[(1, 0)]
                    new_im.paste(ship_im, (i_im, j_im))
                elif positions[1][0] == (i, j):
                    ship_im = self.position_images_[(2, 0)]
                    new_im.paste(ship_im, (i_im, j_im))
                else:
                    new_im.paste(im, (i_im, j_im))

                for player in range(1, 3):
                    pir_images = []
                    for pir in range(1, 4):
                        if positions[player - 1][pir] == (i, j):
                            pir_im = self.position_images_[(player, pir)]
                            pir_images.append(pir_im)
                    if len(pir_images) == 1:
                        pir_im = pir_images[0]
                        new_im.paste(pir_im, (i_im + self.pir_size, j_im + self.pir_size), pir_im)
                    elif len(pir_images) == 2:
                        pir_im1 = pir_images[0]
                        new_im.paste(pir_im1, (i_im + int(self.pir_size/2), j_im + self.pir_size), pir_im1)
                        pir_im2 = pir_images[1]
                        new_im.paste(pir_im2, (i_im + int(3*self.pir_size/2), j_im + self.pir_size), pir_im2)
                    elif len(pir_images) == 3:
                        pir_im1 = pir_images[0]
                        new_im.paste(pir_im1, (i_im + self.pir_size, j_im + int(self.pir_size/2)), pir_im1)
                        pir_im2 = pir_images[1]
                        new_im.paste(pir_im2, (i_im + int(self.pir_size/2), j_im + int(3*self.pir_size/2)), pir_im2)
                        pir_im3 = pir_images[2]
                        new_im.paste(pir_im3, (i_im + int(3*self.pir_size/2), j_im + int(3*self.pir_size/2)), pir_im3)

                gold = int(gold_field[i, j])
                if gold > 0:
                    draw.text((i_im, j_im), f'{gold}', self.rgb_orange, font=self.gold_font)

                if i == (field.shape[0]-1) and j == (field.shape[1]-1):
                    draw.text((i_im, j_im), f'Turn: {turn_count}', self.rgb_red, font=self.info_font)
                    draw.text((i_im, j_im+int(self.step/4)), f'Gold left: {gold_left}',
                              self.rgb_red, font=self.info_font)
                    first_gold, second_gold = gold_counts
                    draw.text((i_im, j_im+int(2 * self.step/4)), f'1 gold: {first_gold}',
                              self.rgb_red, font=self.info_font)
                    draw.text((i_im, j_im+int(3 * self.step/4)), f'2 gold: {second_gold}',
                              self.rgb_red, font=self.info_font)
        new_im = new_im.resize(size)
        return new_im
