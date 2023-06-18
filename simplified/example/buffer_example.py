from simplified.util.buffer import ReplayBuffer
import numpy as np

buffer_size = 10
buffer = ReplayBuffer(buffer_size)
for i in range(15):
    buffer.add((i, np.array([2, 3, 4, i])))
print(buffer)
print(buffer.sample(3))
