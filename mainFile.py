# Tutorial being followed: https://www.youtube.com/watch?v=V_xro1bcAuA&t=2598s

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

# Scalar
scalar = torch.tensor(7)

print(scalar.ndim)

print(scalar.item())

# Vector
vector = torch.tensor([7,7])

vector.shape

# Matrix
MATRIX = torch.tensor([[7,8],
                       [9,10]])
print(MATRIX)
print(MATRIX[1,0]) # Find the second row, first column element.
print(MATRIX.ndim)

# Tensor
TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
print(TENSOR)
print(TENSOR.ndim) # 3 dimensions
print(TENSOR.shape) # 1 x 3 x 3 Dimension.
print(TENSOR[0,1,0]) # Find within the first tensor, the second row, first column element.

# Create random tensor of size (3,4)

random_tensor = torch.rand(3,4)
print(random_tensor)
print(random_tensor.ndim)

# Create random tensor of size similar shape to image tensor
random_image_size_tensor = torch.rand(size=(3,224,224)) # Colour Channels (R,G,B), Height, Width
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

# Zeroes and Ones
zero = torch.zeros(size=(3,4))
print(zero)