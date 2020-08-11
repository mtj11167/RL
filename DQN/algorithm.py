import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQN

class Algorithm():
    def __init__(self, lr, gamma, act_dim, state_dim, memory_capacity, epsilon, batch_size):
        self.model = DQN(state_dim, act_dim)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory_capacity = memory_capacity
        self.replay_buffer = np.zeros((memory_capacity, 2*state_dim+3))
        self.memory_counter = 0
        self.batch_size = batch_size
    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def pridict(self, obs):
        return self.model.forward(obs)
    def choose_action(self, state):
        state = torch.unsqueeze(torch.Tensor(state),0)
        if np.random.rand() <= self.epsilon:
            action_value = self.model.forward(state)
            action = torch.max(action_value,dim=1)[1].numpy()[0]
        else:
            action = np.random.randint(0, self.act_dim)
        return action
    def store_transition(self, state, action, reward, next_state, done):
        transition = np.hstack((state, [action, reward], next_state,done))
        index = self.memory_counter % self.memory_capacity
        self.replay_buffer[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.replay_buffer[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
        batch_action = torch.LongTensor(batch_memory[:, self.state_dim:self.state_dim + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.state_dim + 1:self.state_dim + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, self.state_dim+2:2*self.state_dim+2])
        batch_done = torch.FloatTensor(batch_memory[:, -1:])

        next_value = self.target_model.forward(batch_next_state)
        max_value = torch.max(next_value,dim=1)[0]
        torch.detach(max_value)

        target = batch_reward.squeeze()+self.gamma*(1-batch_done).squeeze()*max_value

        q_value = self.model.forward(batch_state)
        behavior = torch.gather(q_value, dim=1,index=batch_action).squeeze()


        self.optimizer.zero_grad()

        output = self.loss(behavior,target)
        output.backward()
        self.optimizer.step()

        return output
