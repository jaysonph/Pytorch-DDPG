import torch
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hid_size=128):
        '''
        Args:
        - s_dim (int): dim of state
        - a_dim (int): dim of action
        - hid_size (int): hidden dim
        '''
        super().__init__()
        self.fc1 = nn.Linear(s_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size//2)
        self.fc3 = nn.Linear(hid_size//2, hid_size//4)
        self.fc4 = nn.Linear(hid_size//4, a_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(3e-3)

        # for l in [self.fc1, self.fc2]:
        #     nn.init.xavier_uniform_(l.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.constant_(l.bias.data, 0)

        # nn.init.xavier_uniform_(self.fc3.weight.data, gain=nn.init.calculate_gain('tanh'))
        # nn.init.constant_(self.fc3.bias.data, 0)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, inp):
        '''
        Args:
        - inp (torch.float32): state tensor [bs, s_dim]
        
        Returns:
        - out (torch.float32): vector of action_dim between [-1,1]
        '''
        out = self.fc1(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.tanh(out)
        return out

class CriticNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hid_size=128):
        '''
        Args:
        - s_dim (int): dim of state
        - a_dim (int): dim of action
        - hid_size (int): hidden dim
        '''
        super().__init__()
        self.fc1s = nn.Linear(s_dim, hid_size//2)
        self.fc1a = nn.Linear(a_dim, hid_size//2)
        self.fc2 = nn.Linear(hid_size, hid_size//2)
        self.fc3 = nn.Linear(hid_size//2, hid_size//4)
        self.fc4 = nn.Linear(hid_size//4, 1)
        self.relu = nn.ReLU()
        self.init_weights(3e-3)

        # for l in [self.fc1s, self.fc1a, self.fc2]:
        #     nn.init.xavier_uniform_(l.weight.data, gain=nn.init.calculate_gain('relu'))
        #     nn.init.constant_(l.bias.data, 0)

        # nn.init.uniform(self.fc3.weight.data, -3e-3, 3e-3)
        # nn.init.constant(self.fc3.bias.data, 0)
    
    def init_weights(self, init_w):
        self.fc1s.weight.data = fanin_init(self.fc1s.weight.data.size())
        self.fc1a.weight.data = fanin_init(self.fc1a.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, s, a):
        s = self.fc1s(s)
        s = self.relu(s)
        a = self.fc1a(a)
        a = self.relu(a)
        # debug()
        out = self.fc2(torch.cat([s,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out