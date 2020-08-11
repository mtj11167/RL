from  algorithm import Algorithm
import gym
import matplotlib.pyplot as plt
class agent():
    def __init__(self,env, act_dim, state_dim, memory_capacity, epsilon, update_target):
        self.env = env
        self.algo = Algorithm(0.0001, 0.99, act_dim, state_dim, memory_capacity, epsilon, 64)
        self.memory_capacity = memory_capacity
        self.update_target = update_target
    def learn(self, epoch):
        reward_list = []
        plt.ion()
        fig, ax = plt.subplots()

        for i in range(epoch):

            state = self.env.reset()
            ep_reward = 0
            while True:
                self.env.render()
                action = self.algo.choose_action(state)
                next_state, reward, done, _=self.env.step(action)
                ep_reward+=reward
                self.algo.store_transition(state, action, reward, next_state,done)
                if(self.algo.memory_counter >= self.memory_capacity):
                    self.algo.learn()
                    if epoch % self.update_target == 0:
                        self.algo.sync_target()
                    if done:
                        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                if done:
                    break
                state = next_state
            reward_list.append(ep_reward)
            ax.set_xlim(0, epoch)
            ax.plot(reward_list, 'g-', label='total_loss')
            plt.pause(0.001)

        self.env.close()

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agt = agent(env, env.action_space.n, env.observation_space.shape[0],100, 0.9,20)
    agt.learn(400)