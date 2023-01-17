import os
import pandas as pd

import cv2
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

path = 'words'
df = pd.read_csv('up_words.csv')
#df = df.drop(4152)
#df = df.drop(113621)
df = df.drop(45370)
#df = df.drop(113621)
df = df.reset_index(drop = True)


print(len(df))

vocabulary = sorted(list(set(char for label in df['text'].values for char in label)))

len_vocab = len(vocabulary) + 1

vocab = dict(enumerate(vocabulary))
int2char = {k:v for k, v in vocab.items()}
char2int = {v:k for k, v in int2char.items()}

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
        self.softmax = nn.LogSoftmax(dim=2)

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


model1 = torch.load('md_f.pth')
optimizer = torch.optim.Adam(model1.parameters(), lr = 1e-5)
optimizer.load_state_dict(torch.load('optimizer.pth'))
model1.to(DEVICE)
#model1.train()
model1.eval()

igg = cv2.imread('tt12.png', cv2.IMREAD_GRAYSCALE)
igg = cv2.resize(igg,(128, 32), interpolation = cv2.INTER_AREA)
igg = igg/255
igg = np.expand_dims(igg, axis = 0)
igg = np.expand_dims(igg, axis = 0)
print(igg.shape)
igg = torch.as_tensor(igg, dtype = torch.float32)
igg = igg.to(DEVICE)
pre = model1(igg)
print(pre)

pre1 = pre.reshape((pre.shape[1], pre.shape[2]))
pre2 = 10**pre1

ll = []
for item in pre2:
  val, idx = torch.max(item, dim = 0)
  ll.append(int(idx.cpu().numpy()))

print(ll)
ans = []
for i in ll:
  if i == 26:
    continue;
  ans.append(int2char[i])
print(ans)