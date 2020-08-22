from convolution_layer_torch import NormalConvolution2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

class ConvolutionModel(nn.Module):
  def __init__(self):
    super(ConvolutionModel, self).__init__()
    self.conv1 = NormalConvolution2D(filters=64, kernel_size=(3, 3), input_size=(3, 32, 32), activation=F.relu)
    conv1_output = (self.conv1.output_size[0], self.conv1.output_size[1] // 2, self.conv1.output_size[2] // 2)

    self.conv2 = NormalConvolution2D(filters=64, kernel_size=(3, 3), input_size=conv1_output, activation=F.relu)
    conv2_output = (self.conv2.output_size[0], self.conv2.output_size[1] // 2, self.conv2.output_size[2] // 2)

    self.pool = nn.MaxPool2d(2, 2)

    self.linear_input = conv2_output[0] * conv2_output[1] * conv2_output[2]
    self.linear1 = nn.Linear(self.linear_input, 120)
    self.linear2 = nn.Linear(120, 84)
    self.linear3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(self.conv1.forward(x))
    x = self.pool(self.conv2.forward(x))
    x = x.reshape(-1, self.linear_input)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x