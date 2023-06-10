import tkinter as tk
from PIL import ImageTk


class GameViewer:
    """
    Class that creates viewer of the game frames using Tkinter
    """
    def __init__(self,
                 map_visualizer,
                 game_progress,
                 viewer_shape=(1920, 1080),
                 img_shape=(700, 700)):
        """
        :param map_visualizer: MapVisualizer, initialized visualizer
        :param game_progress: list(tuple(np.array, np.array, np.array, tuple(list, list), int, int, tuple(int, int))
        masked_field, full_field, gold_field, positions, turn_count, gold_left, gold_counts
        :param viewer_shape: tuple(int, int), shape of viewer app
        :param img_shape: tuple(int, int), shape of frames
        """
        self.map_visualizer = map_visualizer
        self.game_progress = game_progress
        self.viewer_shape = viewer_shape
        self.img_shape = img_shape
        masked_imgs = []
        full_imgs = []
        for elem in self.game_progress:
            masked_field, full_field, gold_field, positions, turn_count, gold_left, gold_counts = elem
            masked_img = self.map_visualizer.draw_field(masked_field, gold_field, positions, turn_count,
                                                        gold_left, gold_counts, self.img_shape)
            full_img = self.map_visualizer.draw_field(full_field, gold_field, positions, turn_count,
                                                      gold_left, gold_counts, self.img_shape)
            masked_imgs.append(masked_img)
            full_imgs.append(full_img)
        self.gui = tk.Tk()
        self.gui.geometry(f'{self.img_shape[0]}x{self.img_shape[1]}')
        masked_imgs_ = [ImageTk.PhotoImage(img) for img in masked_imgs]
        full_imgs_ = [ImageTk.PhotoImage(img) for img in full_imgs]
        self.image_lists = [masked_imgs_, full_imgs_]
        self.canvas = tk.Canvas(self.gui, width=self.img_shape[0], height=self.img_shape[1])
        self.curr_image = self.canvas.create_image(0, 0, anchor='nw', image=self.image_lists[0][0])
        self.canvas.grid(row=2, column=1)
        self.gui.bind('<Key>', self.update_img)
        self.list_index, self.img_index = 0, 0

    def update_img(self, event):
        """
        Update image on keyboard buttons clicks

        :param event: None
        :return: None
        """
        keysym = event.keysym
        if keysym == 'Left':
            self.img_index = max(self.img_index - 1, 0)
            self.canvas.itemconfig(self.curr_image, image=self.image_lists[self.list_index][self.img_index])
        elif keysym == 'Right':
            self.img_index = min(self.img_index + 1, len(self.image_lists[0]) - 1)
            self.canvas.itemconfig(self.curr_image, image=self.image_lists[self.list_index][self.img_index])
        elif keysym == 'Down':
            self.list_index = max(self.list_index - 1, 0)
            self.canvas.itemconfig(self.curr_image, image=self.image_lists[self.list_index][self.img_index])
        elif keysym == 'Up':
            self.list_index = min(self.list_index + 1, len(self.image_lists) - 1)
            self.canvas.itemconfig(self.curr_image, image=self.image_lists[self.list_index][self.img_index])

    def start(self):
        """
        Initialize app

        :return: None
        """
        self.gui.mainloop()
