import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, image_shape):
        super(CNNModel, self).__init__()
        self.seed = 0
        self.conv1 = nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(512)


        # flat_size = 128 * (image_shape[1] // (3 ** 4)) * (image_shape[2] // (3 ** 4))
        flat_size = 512 * 1 * 1
        self.fc1 = nn.Linear(flat_size, 1024)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        x = self.maxpool4(F.relu(self.bn4(self.conv4(x))))
        x = self.maxpool5(F.relu(self.bn5(self.conv5(x))))
        x = self.maxpool6(F.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


