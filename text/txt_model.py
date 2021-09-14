"""
    Name: txt_model.py
    
    Description: Definition of a text model
                 that takes BERT embeddings as input
                 and outputs emotion classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TxtModel(nn.Module):
    """ Text model """
    def __init__(self):
        super(TxtModel, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(1024, 256,
                                    kernel_size=1, stride=1,
                                    padding=1, dilation=1,
                                    bias=True)
        self.conv2 = torch.nn.Conv1d(256, 256,
                                    kernel_size=1, stride=1,
                                    padding=1, dilation=1,
                                    bias=True)
        self.conv3 = torch.nn.Conv1d(256, 128,
                                    kernel_size=8, stride=1,
                                    padding=1, dilation=1,
                                    bias=True)
        self.conv4 = torch.nn.Conv1d(128, 64,
                                    kernel_size=4, stride=1,
                                    padding=1, dilation=1,
                                    bias=True)
        
        self.mean_pool = nn.AvgPool1d(118)

        self.fc = nn.Linear(64,4)

    def forward(self, x):
        x = x.transpose(1,2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.mean_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = torch.sigmoid(self.fc(x))
        return x