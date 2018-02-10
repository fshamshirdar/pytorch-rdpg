import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20, 50)
        self.lstm = nn.LSTMCell(50, 50)
        self.fc3 = nn.Linear(50, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

        self.cx = Variable(torch.zeros(1, 50)).type(FLOAT)
        self.hx = Variable(torch.zeros(1, 50)).type(FLOAT)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = Variable(torch.zeros(1, 50)).type(FLOAT)
            self.hx = Variable(torch.zeros(1, 50)).type(FLOAT)
        else:
            self.cx = Variable(self.cx.data).type(FLOAT)
            self.hx = Variable(self.hx.data).type(FLOAT)

    def forward(self, x, hidden_states=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if hidden_states == None:
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.lstm(x, hidden_states)

        x = hx
        x = self.fc3(x)
        x = self.tanh(x)
        return x, (hx, cx)

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20 + nb_actions, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        #out = self.fc2(torch.cat([out,a],dim=1)) # dim should be 1, why doesn't work?
        out = self.fc2(torch.cat([out,a], 1)) # dim should be 1, why doesn't work?
        out = self.relu(out)
        out = self.fc3(out)
        return out
