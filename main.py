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

#from data_process import

path = 'words/Words/'
df = pd.read_csv('up_wd1.csv')
#df = df.drop(45370)
#df = df.drop(113621)
#df = df.reset_index(drop = True)

print(len(df))


vocabulary = sorted(list(set(char for label in df['text'].values for char in label)))

len_vocab = len(vocabulary) + 1

vocab = dict(enumerate(vocabulary))
int2char = {k:v for k, v in vocab.items()}
char2int = {v:k for k, v in int2char.items()}

max_len = 0;
for item in df['text'].values:
  max_len = max(max_len, len(item))

print("max word length: ", max_len)
print("vocab length: ", len_vocab)

def encoder(st):
  token = torch.tensor([char2int[i] for i in st])
  token = F.pad(token, pad = (0, max_len-len(token)), mode = 'constant', value = len_vocab - 1)
  return token

def decoder(token):
  token = token[token != len_vocab - 1]
  st = [int2char[i.item()] for i in token]
  return "".join(st)


class HRDataset(Dataset):
  def __init__(self, data, IMG_HEIGHT, IMG_WIDTH):
    #super().__init__()
    self.data = data
    self.IMG_HEIGHT = IMG_HEIGHT
    self.IMG_WIDTH = IMG_WIDTH

  def __len__(self):
    return self.data.shape[0]


  def __getitem__(self, index):
    img_path = self.data.iloc[index]['path']
    img = cv2.imread(path + '/' + img_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation = cv2.INTER_AREA)
    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    #img = cv2.transpose(img)
    img = img/255
    img = np.expand_dims(img, 0)

    label = self.data.iloc[index]['text']
    label = encoder(label)

    img = torch.as_tensor(img, dtype = torch.float32)
    label = torch.as_tensor(label, dtype = torch.int32)

    return img, label


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


criterion = nn.CTCLoss(blank = len_vocab - 1, reduction = 'mean', zero_infinity = True)
model1 = torch.load('md_f.pth')
#model1 = Overall(64, 128, len_vocab, 2)
optimizer = torch.optim.Adam(model1.parameters(), lr = 1e-3)
optimizer.load_state_dict(torch.load('optimizer.pth'))
model1.to(DEVICE)
model1.train()

b_size = 128
#b_size = 18;
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
#txt = ['test10.tif', 'test11.tif', 'test12.tif', 'test13.tif', 'test16.tif', 'test17.tif', 'test18.tif', 'test19.tif', 'test20.png', 'test21.png', 'tt5.png', 'tt6.png', 'tt7.png', 'tt8.png', 'tt9.png', 'tt10.png', 'tt11.png', 'tt12.png']
#ll = ['vamstions', 'kesariyamp', 'hariramjib', 'twmbahuz', 'abcdefgh', 'ijklmnop', 'qrstuvwx', 'tensinbeo', 'abcdezon', 'sunotmeya','kesar', 'vocbam', 'industry', 'amantio', 'cfcermtb', 'vidhyan', 'abcdefg', 'kejilac']

#df = pd.DataFrame()
#df['path'] = txt;
#df['text'] = ll;
dataset = HRDataset(data = df, IMG_HEIGHT = IMAGE_HEIGHT, IMG_WIDTH = IMAGE_WIDTH)
loader = DataLoader(dataset, batch_size=b_size, shuffle=True, pin_memory=True)

from tqdm import tqdm
#from itertools import groupby

# from perc import Perc
# from progiter import ProgIter

for i in range(10):
    print(i)
    loop = tqdm(loader)
    total_loss = 0;
    total = 0
    correct = 0
    for idx, (inputs, labels) in enumerate(loop):
        b_size = inputs.shape[0]
        inputs = inputs.to(DEVICE)
        y_pred = model1(inputs)
        y_pred = y_pred.permute(1, 0, 2).contiguous().requires_grad_(True)

        # print(y_pred.cpu()[1])
        # labels = labels[labels != 0]
        input_lengths = torch.IntTensor(b_size).fill_(16)  # len of output from overall model =  32

        ll = []
        for itm in labels:
            #itm = itm[itm != len_vocab - 1]
            ll.append(len(itm[itm != len_vocab - 1]))

        #target_lengths = torch.IntTensor(ll)
        # target_lengths = torch.count_nonzero(labels, axis=1)
        loss = criterion(y_pred.cpu(), labels, input_lengths, torch.IntTensor(ll))
        total_loss += loss.detach().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model1.state_dict(), 'md_w_f.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')
    torch.save(model1, 'md_f.pth')
    # ratio = correct / total
    # print('TEST correct: ', correct, '/', total, ' P:', ratio)
    print("Avg CTC loss:", total_loss / idx)

torch.save(model1.state_dict(), 'md_w3_f.pth')
torch.save(optimizer.state_dict(), 'optimizer.pth')
torch.save(model1, 'md_f.pth')