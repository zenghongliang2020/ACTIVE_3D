import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Dueling_net(nn.Module):
    def __init__(self, lr, state_dim, action_1_dim, action_2_dim):
        super(Dueling_net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.V = nn.Linear(256, 1)
        self.A_1 = nn.Linear(256, action_1_dim)
        self.A_2 = nn.Linear(256, action_2_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))

        V = self.V(x)
        A_1 = self.A_1(x)
        A_2 = self.A_2(x)

        return V, A_1, A_2


