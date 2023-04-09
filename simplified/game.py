class SimpleGame:
    """
    Class that encapsulates simplified Jackal game logic
    """
    def __init__(self,
                 field):
        """
        :param field: np.array, actual field that is hidden for players
        """
        self.field = field
