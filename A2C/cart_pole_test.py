import gym
from RL.Algorithms.mA2C import *
import matplotlib.pyplot as plt
eps = np.finfo(np.float32).eps.item()
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

use_gpu = False
if torch.cuda.is_available():
    use_gpu = True

class Agent(object):
    def __init__(self,env,obs_dim,action_dim,hidden,gamma,lr,epesode_max,trian_steps):
        super(Agent,self).__init__()
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.trian_steps = trian_steps
        self.gamma = gamma
        self.lr = lr
        self.episode_max =epesode_max
        self.A2C = A2C(self.obs_dim,self.action_dim,self.hidden,self.gamma,self.lr,True)

    def get_target_value(self,value,rewards,mask):
        R = value
        ret = []
        for idx in range(len(rewards)-1,-1,-1):
            R = rewards[idx]+self.gamma*R*mask[idx]
            ret.insert(0,R)
        return ret

    def train(self):
        living_time = []
        for i_episode in range(self.episode_max):
            t=0
            obs = self.env.reset()
            r_epi = []
            c_losses = []
            a_losses = []

            log_probs = []
            rewards = []
            values = []
            done_mask = []
            while True:
                if self.trian_steps < 0:
                    self.env.render()
                    dist = self.A2C.sample(torch.unsqueeze(torch.FloatTensor(obs),0))
                    action = dist.sample()
                    obs_new,r,done,_ = self.env.step(action.numpy()[0])
                    log_probs.append(dist.log_prob(action).unsqueeze(0))
                    rewards.append(torch.FloatTensor([r]))
                    values.append(self.A2C.critic(torch.unsqueeze(torch.FloatTensor(obs),0)))
                    done_mask.append(torch.FloatTensor([1.0-done]))
                    r_epi.append(r)
                    obs = obs_new
                    t+=1
                    if done:
                        obs_new = torch.unsqueeze(torch.FloatTensor(obs_new),0)
                        obs_new_value = self.A2C.critic(obs_new)
                        target_value = self.get_target_value(obs_new_value,rewards,done_mask)
                        c_loss,a_loss = self.A2C.udpate(torch.cat(target_value),torch.cat(values),torch.cat(log_probs))
                        c_losses.append(c_loss)
                        a_losses.append(a_loss)
                        print("episode %i, rewards:%f, actor loss:%f, critic loss%f"
                            %(i_episode,sum(r_epi),np.mean(a_losses),np.mean(c_losses)))
                        r_epi = []
                        c_losses = []
                        a_losses = []
                        living_time.append(t)
                        plot(living_time)
                        break
                else:
                    for _ in range(self.trian_steps):
                        # self.env.render()
                        dist = self.A2C.sample(torch.unsqueeze(torch.FloatTensor(obs),0))
                        action = dist.sample()
                        obs_new,r,done,_ = self.env.step(action.cpu().numpy()[0])
                        log_probs.append(dist.log_prob(action).unsqueeze(0))
                        rewards.append(torch.FloatTensor([r]))
                        values.append(self.A2C.critic(torch.unsqueeze(torch.FloatTensor(obs),0)))
                        done_mask.append(torch.FloatTensor([1.0-done]))
                        r_epi.append(r)
                        obs = obs_new
                        t+=1
                        if done:
                            break
                    obs_new = torch.unsqueeze(torch.FloatTensor(obs_new),0)
                    obs_new_value = self.A2C.critic(obs_new)

                    if use_gpu:
                        rewards = torch.tensor(rewards).cuda()
                        done_mask = torch.tensor(done_mask).cuda()
                    target_value = self.get_target_value(obs_new_value,rewards,done_mask)
                    # target_value = torch.tensor(target_value)
                    # target_value = (target_value - target_value.mean()) / (target_value.std()+eps)
                    # c_loss,a_loss = self.A2C.udpate(target_value.unsqueeze(1),torch.cat(values),torch.cat(log_probs))
                    c_loss,a_loss = self.A2C.udpate(torch.cat(target_value),torch.cat(values),torch.cat(log_probs))
                    c_losses.append(c_loss.cpu())
                    a_losses.append(a_loss.cpu())
                    log_probs = []
                    rewards = []
                    values = []
                    done_mask = []
                    if done:
                        print("episode %i, rewards:%f, actor loss:%f, critic loss%f"
                            %(i_episode,sum(r_epi),np.mean(a_losses),np.mean(c_losses)))
                        r_epi = []
                        c_losses = []
                        a_losses = []
                        living_time.append(t)
                        # plot(living_time)
                        break
            # if i_episode % 100 ==0:
            #     print("#########evaluate########")
            #     r_epi_all = []
            #     for i_all in range(10):
            #         obs = self.env.reset()
            #         r_epi = []
            #         while True:
            #             self.env.render()
            #             action = self.A2C.pridict(torch.unsqueeze(torch.FloatTensor(obs),0))
            #             obs_new,r,done,_ = self.env.step(action.numpy()[0])
            #             r_epi.append(r)
            #             if done:
            #                 print("ecaluate episode %i, rewards:%f"%(i_all,sum(r_epi)))
            #                 r_epi_all.append(sum(r_epi))
            #                 r_epi = []
            #                 break
            #             obs = obs_new
            #     print("ecaluate, rewards anverage:%f"%(np.mean(r_epi_all)))
            #     print("#########evaluate over########")
                    

    def predict(self,obs):
        return self.A2C.pridict(obs)
    
    def sample(self,obs):
        return self.A2C.sample(obs).sample()
    def value(self,obs):
        return self.A2C.critic(obs)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)

    # path = './AC_CartPole-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    # if len(steps) % 200 == 0:
    #     plt.savefig(path)
    plt.pause(0.0000001)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden = [32]
    episodes = 1000
    trian_steps = 50
    agent = Agent(env,obs_dim,action_dim,hidden,0.99,0.01,episodes,trian_steps)
    agent.train()