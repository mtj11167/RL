import tensorflow as tf
import gym
import numpy as np

def generate_NN(x,sizes,out_activation=None,activation=tf.tanh):
    '''
    :param x: input的张量
    :param sizes:list 有隐藏层和输出层的大小按顺序组成
    :param out_activation:隐藏层的激活函数
    :param activation:输出层的激活函数
    :return:返回最终网络的张量
    '''
    for size in sizes[:-1]:
        x=tf.layers.dense(x,units=size,activation=activation)
    return tf.layers.dense(x,units=sizes[-1],activation=out_activation)

def train(env_name='CartPole-v0', hidden_layer=None,lr=1e-2,epochs=50,batch_size=5000,render=False):
    if hidden_layer is None:
        hidden_layer = [32]
    env=gym.make(env_name)
    obs_n=env.observation_space.shape[0]
    actions_n=env.action_space.n

    obs_ph=tf.placeholder(shape=(None, obs_n), dtype=tf.float32)
    nn=generate_NN(obs_ph, hidden_layer + [actions_n])
    action=tf.squeeze(tf.multinomial(logits=nn,num_samples=1),axis=1)


    weight_ph=tf.placeholder(shape=(None,),dtype=tf.float32)
    action_ph=tf.placeholder(shape=(None,),dtype=tf.int32)
    action_mask=tf.one_hot(indices=action_ph,depth=actions_n)
    log_pro=tf.reduce_sum(action_mask*tf.nn.log_softmax(nn),axis=1)
    loss=-tf.reduce_mean(weight_ph*log_pro)
    train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def one_epoch():
        batch_weight=[]
        batch_obs=[]
        batch_actions=[]
        rws=[]
        returns=[]

        done=False

        finished_rendering_this_epoch = False
        obs = env.reset()

        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())
            act=sess.run(action, feed_dict={obs_ph:obs.reshape(1, -1)})[0]
            batch_actions.append(act)
            obs,reward,done,_=env.step(act)
            rws.append(reward)


            if done:
                # print("one episode over")
                ret=sum(rws)
                batch_weight+=([ret]*len(rws))
                returns.append(ret)

                rws=[]
                done=False
                obs = env.reset()

                finished_rendering_this_epoch = True

                if len(batch_weight) > batch_size:
                    break
        epoch_loss,_ = sess.run([loss, train_op],
                              feed_dict={weight_ph: np.array(batch_weight), action_ph: np.array(batch_actions),obs_ph: np.array(batch_obs)})
        return epoch_loss,returns

    for i in range(epochs):
        epochloss,epoch_return=one_epoch()
        print(i," the loss is: ",epochloss," return is: ",np.mean(epoch_return))

if __name__ == "__main__":
    train()

