import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from convolution_network_torch import ConvolutionModel
import pyprind

def train(model, criterion, optimizer, epochs, trainloader, testloader, device):
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    bar = pyprind.ProgBar(len(trainloader), track_time=True, title="Training Model")
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        bar.update()

    print("Epoch: {} Loss: {:.3f}".format(epoch + 1, running_loss / len(trainloader)))

    correct = 0
    total = 0

    with torch.no_grad():
      for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

  print("Finished Training")
  return model

def get_cifar10(batch_size):
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=6)

  testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=6)

  return trainloader, testloader

device = torch.device("cuda")
model = ConvolutionModel()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
trainloader, testloader = get_cifar10(64)
train(model, criterion, optimizer, 50, trainloader, testloader, device)
