import torch
import torch.nn as nn
import numpy as np
import copy
from replay_buffer import ReplayBuffer, Transition
from model import ActorNetwork, CriticNetwork

class DDPG():
    def __init__(self, capacity, s_dim, a_dim, a_low_bound, a_up_bound, tau, noise_std_min, noise_std_max, noise_decay_steps, 
                 gamma, actor_lr, critic_lr, hid_size, batch_size, target_update_intv, start_steps, device):
        
        self.a_dim = a_dim
        self.a_low_bound = a_low_bound
        self.a_up_bound = a_up_bound
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_intv = target_update_intv
        self.start_steps = start_steps
        self.device = device

        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_decay_steps = noise_decay_steps
        self.noise_std = -1
        self.step = 0
        self.train_iter = 0
        self.train = True

        self.replay_buffer = ReplayBuffer(capacity)

        self.actor_net = ActorNetwork(s_dim, a_dim, hid_size).to(device)
        self.critic_net = CriticNetwork(s_dim, a_dim, hid_size).to(device)
        # self.actor_net = Actor().to(device)
        # self.critic_net = Critic().to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        for param in self.target_actor_net.parameters():
            param.requires_grad = False

        for param in self.target_critic_net.parameters():
            param.requires_grad = False

        self.critic_loss_fn = nn.MSELoss()
        # self.critic_loss_fn = nn.SmoothL1Loss()

        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr = actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr = critic_lr)

    def act(self, s):
        '''
        Args:
        - s (np.float32): state tensor [s_dim, ]

        Returns:
        - a (np.float32): action tensor [a_dim, ]
        ''' 
        # self.step += 1
        if self.train and self.step < self.start_steps:
            # a = np.random.normal(size=self.a_low_bound.shape)
            a = np.random.uniform(low=self.a_low_bound, high=self.a_up_bound, size=self.a_low_bound.shape)
            a = np.clip(a, self.a_low_bound, self.a_up_bound).astype(np.float32)
        else:
            s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.actor_net.eval()
            with torch.no_grad():
                a = self.actor_net(s).cpu().squeeze(0)
            if self.train:
              self.noise_std = self.noise_std_max - (self.noise_std_max - self.noise_std_min) * self.step / (self.noise_decay_steps + self.start_steps)  # Linear decaying of noise std
              self.noise_std = max(self.noise_std, self.noise_std_min)
              noise = torch.normal(0., self.noise_std, (self.a_dim,))
            else:
              noise = 0
            # noise = torch.normal(0., self.noise_std_max, (self.a_dim,))
            a = torch.clamp(a+noise, torch.tensor(self.a_low_bound), torch.tensor(self.a_up_bound))
            a = np.array(a)
        return a

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
            
        self.train_iter += 1

        self.actor_net.train()
        self.critic_net.train()
        self.target_actor_net.eval()
        self.target_critic_net.eval()

        b_states, b_actions, b_rewards, b_next_states, b_done = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))

        b_done = torch.tensor(b_done, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [bs, 1]
        b_states = torch.tensor(b_states, dtype=torch.float32, device=self.device)  # [bs, s_dim]
        b_actions = torch.tensor(b_actions, device=self.device)  # [bs, a_dim]
        b_rewards = torch.tensor(b_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [bs, 1]
        b_next_states = torch.tensor(b_next_states, dtype=torch.float32, device=self.device)  # [bs, s_dim]
        
        # Update Critic Network
        self.critic_net.train()
        for param in self.critic_net.parameters():
            param.requires_grad = True
        with torch.no_grad():
            a_ = self.target_actor_net(b_next_states)
            y_ = self.target_critic_net(b_next_states, a_).detach()
        y = b_rewards + self.gamma * (1 - b_done) * y_
        q_pred = self.critic_net(b_states, b_actions)
        critic_loss = self.critic_loss_fn(q_pred, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor Network
        self.critic_net.eval()
        for param in self.critic_net.parameters():
            param.requires_grad = False
        a_pred = self.actor_net(b_states)
        actor_loss = - self.critic_net(b_states, a_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.train_iter % self.target_update_intv == 0:
            self.update_target_nets()

        return critic_loss.detach().item(), actor_loss.detach().item()

    def update_target_nets(self, method='soft'):
        '''
        Args:
        - method: 'soft' for soft update OR 'hard' for hard update
        '''
        if method == 'soft':
            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        if method == 'hard':
            self.target_actor_net.load_state_dict(self.actor_net.state_dict())
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())


        for param in self.target_actor_net.parameters():
            param.requires_grad = False

        for param in self.target_critic_net.parameters():
            param.requires_grad = False