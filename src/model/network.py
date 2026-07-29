import torch
import torch.nn as nn

class CifarCNN(nn.Module):
  """A compact, memory-efficient CNN designed to generate complex FP32 distributions."""
  
  def __init__(self):
    super(CifarCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 2)
    
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2, 2)
    
    self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
    self.relu3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2, 2)
    
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(128 * 4 * 4, 256)
    self.relu4 = nn.ReLU()
    self.fc2 = nn.Linear(256, 10)

  def forward(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = self.pool3(self.relu3(self.conv3(x)))
    
    x = self.flatten(x)
    
    x = self.relu4(self.fc1(x))
    x = self.fc2(x)
    return x