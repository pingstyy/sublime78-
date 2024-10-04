import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, num_heads, hidden_dim):
        super(ViT, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=hidden_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(3, 64)  # input layer
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)
        self.avgpool = nn.AvgPool2d(2)
        self.vit1 = ViT(1024, num_heads=8, hidden_dim=256)
        self.vit2 = ViT(1024, num_heads=8, hidden_dim=256)
        self.fc = nn.Linear(1024, 10)  # output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.cat((self.vit1(x), self.vit2(x)), dim=1)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

net = Net()