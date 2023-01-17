import torch.onnx as tonnx
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import pandas as pd

import cv2
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 64)

    def forward(self, x):
        # print(x.shape)
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # print(out.shape)
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))
        # print(out.shape)
        out = self.maxpool(self.relu(self.bn3(self.conv3(out))))
        # print(out.shape)
        out = out.permute(0, 3, 2, 1)
        # print(out.shape)
        out = out.reshape((out.shape[0], out.shape[1], -1))
        # print(out.shape)
        out = torch.stack([self.relu(self.fc(out[i])) for i in range(out.shape[0])])
        # print(out.shape)
        return out


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BLSTM(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super(BLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hid_size
        self.lstm = nn.LSTM(in_size, hid_size, num_layers, batch_first=True, bidirectional=True, dropout=0.25)
        # self.lstm1 = nn.LSTM(256, 64, num_layers, batch_first = True, bidirectional = True, dropout = 0.25)
        self.fc = nn.Linear(hid_size * 2, out_size)
        # self.fc = nn.Linear(hid_size*2, out_size)
        self.softmax = nn.LogSoftmax(dim = 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE)

        # print(" phase 2")
        # print(x.shape)
        outputs, hidden = self.lstm(x, (h0, c0))
        # print(outputs.shape)

        # h1 = torch.zeros(self.num_layers * 2, outputs.size(0), 64).to(DEVICE)
        # c1 = torch.zeros(self.num_layers * 2, outputs.size(0), 64).to(DEVICE)
        # outputs, hidden = self.lstm1(outputs, (h1, c1))
        # print(outputs.shape)
        outputs = torch.stack([self.fc(outputs[i]) for i in range(outputs.shape[0])])
        # print(outputs.shape)
        outputs = self.softmax(outputs)

        return outputs


class Overall(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super(Overall, self).__init__()
        self.blstm = BLSTM(in_size, hid_size, out_size, num_layers)
        self.fe = CNN()

    def forward(self, x):
        out = self.fe(x)
        out = self.blstm(out)
        return out

dummy_input = torch.randn(1, 1, 32, 128, requires_grad=True)
model = torch.load('md_f.pth')
model.eval()
tonnx.export(model, dummy_input, 'torch_onnx.onnx', opset_version=14)