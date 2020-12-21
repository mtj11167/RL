import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden,use_gpu):
        super(Actor,self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.use_gpu = use_gpu
        self.nets_dim = [self.obs_dim]+hidden+[self.action_dim]
        self.nets = nn.ModuleList() 
        self.init_nets()
    
    def init_nets(self):
        for idx in range(len(self.nets_dim)-1):
            self.nets.append(nn.Linear(self.nets_dim[idx],self.nets_dim[idx+1]))
    
    def forward(self,obs):
        if self.use_gpu:
            obs = obs.cuda()
        m_obs = obs
        for idx,net in enumerate(self.nets):
            if idx < self.nets.__len__()-1:
                m_obs = F.relu(net(m_obs))
            else:
                m_obs = net(m_obs)
        return m_obs

class Critic(nn.Module):
    def __init__(self,obs_dim,hidden,use_gpu):
        super(Critic,self).__init__()
        self.obs_dim  =obs_dim
        self.use_gpu = use_gpu
        self.nets_dim = [self.obs_dim]+hidden+[1]
        self.nets = nn.ModuleList()
        self.init_nets()
    
    def init_nets(self):
        for idx in range(len(self.nets_dim)-1):
            self.nets.append(nn.Linear(self.nets_dim[idx],self.nets_dim[idx+1])) 
    
    def forward(self,obs):
        if self.use_gpu:
            obs = obs.cuda()
        m_obs = obs
        for idx,net in enumerate(self.nets):
            if idx < self.nets.__len__()-1:
                m_obs = F.relu(net(m_obs))
            else:
                m_obs = net(m_obs)
        return m_obs

class A2C(object):
    def __init__(self,obs_dim,action_dim,hidden,gamma,lr,use_gpu):
        super(A2C,self).__init__()
        self.obs_dim  =obs_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.gamma = gamma
        self.lr = lr
        self.use_gpu = use_gpu
        self.actor = Actor(self.obs_dim,self.action_dim,self.hidden,self.use_gpu)
        self.critic = Critic(self.obs_dim,self.hidden,self.use_gpu)
        if self.use_gpu:
            self.actor.cuda()
            self.critic.cuda()
        # self.critic_loss = nn.MSELoss()
        self.actor_optim = Adam(self.actor.parameters(),lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(),lr=self.lr)

    def pridict(self,obs):
        if self.use_gpu:
            obs = obs.cuda()
        logits = self.actor(obs)
        probs = F.softmax(logits,dim=1)
        return torch.argmax(probs,dim=1)
    
    def sample(self,obs):
        if self.use_gpu:
            obs = obs.cuda()
        logits = self.actor(obs)
        probs = F.softmax(logits,dim=-1)
        # return Categorical(probs.cpu())
        return Categorical(probs)
    
    def udpate(self,target,value,log_probs):
        if self.use_gpu:
            target = target.cuda()
            value = value.cuda()
            log_probs = log_probs.cuda()
        critic_loss_out = (target-value).pow(2).mean()
        # critic_loss_out = self.critic_loss(value,target)
        self.critic_optim.zero_grad()
        critic_loss_out.backward(retain_graph = True)
        self.critic_optim.step()

        td_error = target-value
        actor_loss = -(log_probs*td_error.detach()).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return critic_loss_out.data,actor_loss.data 

    def learn(self,s,a,r,s_):
        target = r+self.gamma*self.critic(s_)
        value = self.critic(s)
        
        critic_loss_out = self.critic_loss(value,target)
        self.critic_optim.zero_grad()
        critic_loss_out.backward()
        self.critic_optim.step()

        td_error = target-value
        dist = Categorical(F.softmax(self.actor(s),dim=-1))
        log_probs = dist.log_prob(a)
        actor_loss = -(log_probs*td_error.detach().mean())
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        return critic_loss_out.data,actor_loss.data

if __name__ == "__main__":
    a = torch.FloatTensor([2,8])
    c = Categorical(logits=a)
    # c = Categorical(probs=a)
    print(a.log())
    print(c.log_prob(torch.tensor([0])))