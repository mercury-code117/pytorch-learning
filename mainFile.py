# Tutorial being followed: https://www.youtube.com/watch?v=V_xro1bcAuA&t=2598s

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### 3 Main errors
# 1. Tensors not right datatype.
# 2. Tensors not right shape.
# 3. Tensors not on the right device.

# Getting info from tensors
# 1. Get the required datatype (dtype) using tensor.dtype
# 2. Get the tensor shape by using tensor.shape
# 3. Get the device a tensor is on by using tensor.device

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
zero = torch.zeros(size=(4,4))

ones = torch.ones(size=(4,4))

# Range of Tensors (and others within the tensor family)
toTen = torch.arange(start = 1, step= 1, end=11)

# Turn the arange() from 1 to 10 into a range of all 0's.
ten_zeros = torch.zeros_like(input=toTen)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # What data type is the tensor (e.g. float32, float16, etc.)
                               device="cuda", # What device is the tensor on
                               requires_grad=False) # Whether or not to track gradients with this tensor's operations.

# Convert the established 32-bit variable into a 16-bit variable.
float_16_tensor = float_32_tensor.type(torch.float16)

# Create a new tensor and find out all of the information.
tensorTest = torch.rand(3,4)

print(tensorTest)
print(f"Datatype of tensor: {tensorTest.dtype}")
print(f"Shape of tensor: {tensorTest.shape}")
print(f"Device tensor is located on: {tensorTest.device}")

# Change device to have it on the gpu
tensorTest = tensorTest.to("cuda")
print(f"Datatype of tensor: {tensorTest.dtype}")
print(f"Shape of tensor: {tensorTest.shape}")
print(f"Device tensor is located on: {tensorTest.device}")

# Add 10 to a tensor (becomes tensor([11, 12, 13]))
tensor = torch.tensor([1,2,3])
tensor + 10

# All tensor values are involved, and other math operations can be done such as multiplication, division and subtraction.
# PyTorch has in-built functions to do this such as:
tensorMul = torch.mul(tensor, 10) # Multiplication
tensorAdd = torch.add(tensor, 10) # Addition
tensorSub = torch.sub(tensor, 10) # Subtraction
tensorDiv = torch.div(tensor, 10) # Division

# Matrix Multiplication
# Two main ways to perform multiplication in neural networks and deep learning:
# 1. Element-wise multiplication.
# 2. Matrix multiplication (Dot product).

# Element-wise multiplication
print(tensor * tensor)

# Matrix multiplication
print(torch.matmul(tensor, tensor))

val = 0
for i in range(len(tensor)):
  val += tensor[i] * tensor[i]
print(val)

# The two examples above is to showcase how an in-built torch function of mathmul() is able to achieve the result
# at a faster time than what doing a for-loop is able to achieve. 