import numpy as np
import torch
import torch.nn as nn

"""
input_size: (3, 20, 20)
kernel_size: (3, 3)
batch_size: 10
filter_number: 6
padding: 0
stride: 1
"""

class NormalConvolution2D(nn.Module):
  def __init__(self, filters, kernel_size, input_size, padding=0, stride=1, activation=torch.sigmoid):
    super().__init__()
    # parameters
    self.kernel_size = kernel_size
    self.filters = filters
    self.padding = padding
    self.stride = stride
    self.activation = activation

    # channels, input image height, input image width
    channels, input_height, input_width = input_size

    # filter size from number of filters, chanels, kernel size
    filter_size = (filters, channels) + kernel_size

    # output height, output width from filter data
    output_height = int(1 + (input_height + 2 * self.padding - self.kernel_size[0]) / self.stride)
    output_width = int(1 + (input_width + 2 * self.padding - self.kernel_size[1]) / self.stride)

    # output size
    self.output_size = (self.filters, output_height, output_width)

    # weight height from filter data (expand filter)
    self.weights_height = self.kernel_size[0] * self.kernel_size[1] * channels

    # layer parameter
    self.weights = nn.Parameter(torch.randn((self.weights_height, self.filters), requires_grad=True))
    self.biases = nn.Parameter(torch.zeros(self.output_size, requires_grad=True))

  def forward(self, x):
    batch_size = x.size()[0]

    # image to column
    unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    x = torch.transpose(unfold(x), 1, 2)
    print(x.size(), self.weights.size())

    # output with matrix multiplication
    output = torch.matmul(x, self.weights)
    output = torch.transpose(output, 1, 2)

    output = output.view((-1,) + self.output_size) + self.biases

    output = self.activation(output)

    return output

image = torch.randn((30, 3, 10, 10))
conv = NormalConvolution2D(filters=8, kernel_size=(2, 2), input_size=(3, 10, 10))
out = conv.forward(image)

pool = nn.MaxPool2d(2, 2)
pool_out = pool(out)
