#coding=utf-8
'''
DnnModel in PyTorch
@author: topzero
'''
import torch.nn as nn
import torch.nn.functional as F


class DnnModel(nn.Module):
    def __init__(self, nbins, nframes, hidden_size=1024):
        super(DnnModel, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(nbins * nframes, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, hidden_size)
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, hidden_size)
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, nbins * nframes)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dnn(x)

