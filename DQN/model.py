import torch
import torch.nn  as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_dim,action_dim):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        action_value = self.fc3(h2)
        return action_value