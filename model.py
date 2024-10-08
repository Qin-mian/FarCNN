import torch
import torch.nn as nn
import torch.nn.functional as F

#软注意力
class SoftAttention(nn.Module):
    def __init__(self, in_channels):
        super(SoftAttention, self).__init__()
        self.in_channels = in_channels
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 使用1x1卷积来学习注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.attention_conv(x)
        attention_weights = self.sigmoid(attention_weights)  # 将权重归一化到0-1之间
        attended = x * attention_weights  # 将注意力权重应用于输入特征
        return attended

class CNNModel(nn.Module):
    def __init__(self, image_shape):
        super(CNNModel, self).__init__()
        self.seed = 0
        self.conv1 = nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=4, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = SoftAttention(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = SoftAttention(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = SoftAttention(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.attention4 = SoftAttention(128)

        # flat_size = 128 * (image_shape[1] // (3 ** 4)) * (image_shape[2] // (3 ** 4))
        flat_size = 128 * 1 * 1
        self.fc1 = nn.Linear(flat_size, 512)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        x = self.maxpool4(F.relu(self.bn4(self.conv4(x))))
        x = self.attention4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


