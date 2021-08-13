import torch
import numpy as np
import gym
import random
random.seed(0)
from agent import DDPG

env = gym.make('LunarLanderContinuous-v2')

n_episodes = 1500
max_steps = 1000

start_steps = 50  # To be handled, in terms of train iterations
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_low_bound = env.action_space.low  # To be handled
a_up_bound = env.action_space.high  # To be handled
tau = 0.001
noise_std_min = 0.0
noise_std_max = 0.8
noise_decay_steps = 200
gamma = 0.99
actor_lr = 0.0001
critic_lr = 0.0001
hid_size = 512
batch_size = 256
target_update_intv = 1
capacity = 100000
test_episodes = 50
test_intv = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = DDPG(capacity, s_dim, a_dim, a_low_bound, a_up_bound, tau, noise_std_min, noise_std_max, noise_decay_steps, 
                 gamma, actor_lr, critic_lr, hid_size, batch_size, target_update_intv, start_steps, device)

r_hist = []
c_loss_hist = []
a_loss_hist = []
for ep in range(n_episodes):
    agent.train = True
    s = env.reset()
    ep_r = 0
    ep_c_loss = 0
    ep_a_loss = 0
    agent.step += 1
    for step in range(max_steps):
        a = agent.act(s)
        s_, r, done, info = env.step(a)
        agent.store_transition(s, a, r, s_, done)
        s = s_
        ep_r += r
        c_loss, a_loss = agent.learn()
        if c_loss is not None and a_loss is not None:
            ep_c_loss += c_loss
            ep_a_loss += -a_loss
        if done:
            break
    print(f'Episode {ep}: reward = {ep_r}')
    r_hist.append(ep_r)
    c_loss_hist.append(ep_c_loss/(step+1))
    a_loss_hist.append(ep_a_loss/(step+1))

    if ep > 0 and ep % test_intv == 0:
        agent.train = False
        test_rewards = []
        for test_ep in range(test_episodes):
            s = env.reset()
            test_ep_r = 0
            for step in range(max_steps):
                a = agent.act(s)
                s, r, done, info = env.step(a)
                test_ep_r += r
                if done:
                    break
            test_rewards.append(test_ep_r)
        avg_reward = np.mean(test_rewards)
        print(f"{'='*20} Evaluation {'='*20}\nAverage Rewards over {test_episodes} episodes = {avg_reward}\n{'='*52}")
        if avg_reward > env.spec.reward_threshold:
            print('************************************************************** Congratulations! Solved **************************************************************')
            break