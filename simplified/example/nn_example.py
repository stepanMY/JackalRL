from simplified.util.nn import ResUnet
import torch
import numpy as np


arry = np.zeros((128, 26, 13, 13), dtype=np.float32)
t = torch.from_numpy(arry)
nnet = ResUnet(26, 48, (13, 13))
nnet.eval()
with torch.no_grad():
    result = nnet(t)
print(result.shape)
