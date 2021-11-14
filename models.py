import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        # Innput image size: 224 * 224 pixels
        # Output image: (W-F)/S + 1 = (224-5)/1 + 1 = 220

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5) # Output: (224-5)/1 + 1 = 220, after pooling 110
        self.conv2 = nn.Conv2d(32, 64, 5) # Output: (110-5)/1 + 1 = 106, after pooling 53
        self.conv3 = nn.Conv2d(64, 128, 3) # Output: (53-3)/1 + 1 = 51, after pooling 25
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, stride=2)

        # dropout layers
        self.dropout1 = nn.Dropout(p=0.5)

        # fully connected layers
        self.fc1 = nn.Linear(128*25*25, 1000)
        self.fc2 = nn.Linear(1000, 136)

        
    def forward(self, x):
        # Define the feedforward behavior of this model

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
