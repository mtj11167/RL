import torch
import gym
a = torch.arange(16).view(4,4)
print(a)
print(a[0])

env = gym.make("CartPole-v0")
env.reset()
print(env.action_space.n)
print(type(env.action_space.sample()))

b=torch.Tensor([0.32])
print(b.numpy()[0])