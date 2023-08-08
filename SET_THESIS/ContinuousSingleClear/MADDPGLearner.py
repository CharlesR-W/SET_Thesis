#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:41:52 2023

@author: crw
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition',
    ('state','act','next_state','reward'),
    )

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class PolicyNet(nn.Module):
    #this class copied from pytorch RL tutorial

    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, n_actions, dtype=torch.float32),
            #nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class QNet(nn.Module):
    #this class copied from pytorch RL tutorial

    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        #breakpoint()
        self.network = nn.Sequential(
        nn.Linear(n_observations + n_actions, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        )
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_vec, act_vec):
        #breakpoint()
        x = torch.cat((state_vec,act_vec),dim=1)
        return self.network(x)

class SingleBidMADDPGLearner():
    
    def __init__(self, param_dict, env, agent_id=None):
        
        self.env = env #just for convenience, keep a handle for the environment
        self.rng = param_dict["rng"] #use the same rng as everyone else for reproducability
        
        self.num_agents = self.env.param_dict["num_agents"]
        
        # Get number of actions from gym action space
        #should be 2x the number of actions, paramd by mu, sigma
        self.my_act_rv_dim = env.get_agent_act_dim() * 2
        self.act_rv_dim = self.my_act_dim * self.num_agents

        # Get the number of state observations
        self.my_state_dim = env.get_agent_state_dim()
        self.state_dim = self.state_dim * self.num_agents

        self.policy_net = PolicyNet(self.my_state_dim, self.my_act_rv_dim).to(device)
        self.Q_net = QNet(self.state_dim, self.act_rv_dim).to(device)

        learning_rate = param_dict["learning_rate"]
        self.critic_optimizer = optim.AdamW(self.Q_net.parameters(), lr=learning_rate, amsgrad=True)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.discount_factor = param_dict["discount_factor"]
        
        self.critic_loss_fn = nn.MSELoss()
        #self.policy_loss_fn = nn.ASDFASDF()
        
        memory_capacity = param_dict["memory_capacity"]
        self.memory = ReplayMemory(memory_capacity)
        
        #self.explore_type = param_dict["explore_type"]
        self.explore_params = param_dict["explore_params"] # should be a dictionary itself
        
        
        self.agent_id = agent_id
        
        self.critic_loss_history = []
        self.policy_loss_history = []
        
    
    def act(self, state, noise=True, remember=True):
        
        #breakpoint()
        if noise:
            state_noise = self.sample_state_noise()
            act_noise = self.sample_act_noise()
            
            state = state + state_noise
        
        act = self.policy_net(state)
        
        if noise:
            act = act + act_noise
        
        if not noise:
            #breakpoint()
            pass
        
        #draw the actions from the parameterized distn; output is [mu1,mu2,sig1,sig2]
        mu_idxs = [0,1]; sig_idxs=[2,3]
        mu = act[:,mu_idxs]
        sigma = act[:, sig_idxs]
        torch.abs_(sigma) #take absolute value since sigma must be > 0
        
        ret = torch.distributions.normal.Normal(loc=mu,scale=sigma).rsample()
        
        #Q must be <=Qmax
        Qmax = self.env.agent_Q_maxes[self.agent_id]
        torch.clamp_(ret[0],min=0,max=Qmax)
        
        if remember:
            pass #TODO implement this (or is it done already somewhere?)
        
        return ret
        
    def sample_state_noise(self):
        mu_state_noise = self.explore_params["mu_state_noise"]
        std_state_noise = self.explore_params["std_state_noise"]
        
        state_noise = np.random.normal(loc=mu_state_noise, scale=std_state_noise, size=self.state_dim)
        state_noise = torch.from_numpy(state_noise).to(device=device,dtype=torch.float32)
        return state_noise
    
    def sample_act_noise(self):
        mu_act_noise = self.explore_params["mu_act_noise"]
        std_act_noise = self.explore_params["std_act_noise"]
        
        act_noise = np.random.normal(loc=mu_act_noise, scale=std_act_noise,size=self.act_dim)
        act_noise = torch.from_numpy(act_noise).to(device=device,dtype=torch.float32)
        return act_noise
        
    
    def learn_from_batch(self, batch):
        #breakpoint()
        states = torch.zeros(size=[len(batch),len(batch[0].state)],dtype=torch.float32).to(device)
        acts = torch.zeros(size=[len(batch),len(batch[0].act)],dtype=torch.float32).to(device)
        rewards = torch.zeros(size=[len(batch)],dtype=torch.float32).to(device)
        for lv_tran, tran in enumerate(batch):
            states[lv_tran] = tran.state
            acts[lv_tran] = tran.act
            rewards[lv_tran] = tran.reward
        
        #breakpoint()
        # batch is a batch of transitions sampled randomly
        current_Q = self.Q_net(states, acts)
        target_Q = rewards.unsqueeze(1) #+ self.discount_factor*next_Q # TODO confirm  Q _remains_ detached here
        
        # first, the Q-loss for the critic
        self.critic_optimizer.zero_grad() #TODO Idk if we should zero AFTER getting current_Q
        #MSE of Q(s,a) vs r + g*Q(s',a')
        critic_loss_val = self.critic_loss_fn(current_Q, target_Q)
        critic_loss_val.backward()
        self.critic_optimizer.step()
        
        #record the critic loss
        self.critic_loss_history.append(critic_loss_val.item())
        
        #the policy loss
        self.policy_optimizer.zero_grad()
        #calculate the action the policy would take
        #breakpoint()
        #get the actions
        policy_acts = self.act(states,noise=False, remember=False) #NB remember doesn't do anything yet
        
        policy_loss_val = -1*torch.mean(self.Q_net(states, policy_acts)) #TODO should there be a detach on Q?  I want grad through a, but not through the Q params
        policy_loss_val.backward()
        self.policy_optimizer.step()
        
        #record the policy loss for the batch
        self.policy_loss_history.append(policy_loss_val.item())