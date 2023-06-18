import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.

        :param size: int, max number of transitions to store in the buffer.
                          When the buffer overflows the old memories are dropped.
        """
        self.buffer = []
        self.maxsize = size
        self.next_idx = 0

    def add(self, value):
        """
        Add value to buffer

        :param value: object
        :return: None
        """
        if self.next_idx >= len(self.buffer):
            self.buffer.append(value)
        else:
            self.buffer[self.next_idx] = value
        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        """
        Sample with replacement from buffer

        :param batch_size: int, number of objects to sample
        :return: list, sample objects
        """
        indexes = list(np.random.randint(0, len(self.buffer), batch_size))
        sample = [self.buffer[i] for i in indexes]
        return sample

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return str(self.buffer)
