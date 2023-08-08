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
        self.full_memory = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        self.full_memory.append(Transition(*args))

    def sample(self, batch_size,randomize=True):
        
        if randomize:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)
    
class PolicyNet(nn.Module):
    #this class copied from pytorch RL tutorial

    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 300, dtype=torch.float32),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 300, dtype=torch.float32),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, n_actions, dtype=torch.float32),
            #nn.tanh() #scale back out from [-1,1]
        )
        
        #self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)

def init_weights(w):
    if isinstance(w, nn.Linear):
        torch.nn.init.xavier_uniform_(w.weight, gain=nn.init.calculate_gain('relu'))
        w.bias.data.fill_(0.01)

class QNet(nn.Module):
    #this class copied from pytorch RL tutorial



    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        #breakpoint()
        self.network = nn.Sequential(
        nn.Linear(n_observations + n_actions, 300),
        nn.LayerNorm(300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.LayerNorm(300),
        nn.ReLU(),
        nn.Linear(300, 1),
        )
        #self.network.apply(init_weights)
        
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_vec, my_act_vec, other_act_vec=None):
        #breakpoint()
        if other_act_vec is not None:
            x = torch.cat((state_vec, my_act_vec, other_act_vec),dim=1)
        else:
            x = torch.cat((state_vec, my_act_vec),dim=1)
        x.requires_grad_()
        x = self.network(x)
        return x

class SingleBidLearner():
    
    def __init__(self, param_dict, env, agent_id=None, actor="MADDPG", has_supervisor=False, COURNOT=False, BERTRAND=False):
        
        self.actor = actor
        self.has_supervisor = has_supervisor
        self.COURNOT = COURNOT
        self.BERTRAND = BERTRAND
        
        self.env = env #just for convenience, keep a handle for the environment
        self.rng = param_dict["rng"] #use the same rng as everyone else for reproducability
        
        self.num_agents = self.env.param_dict["num_agents"]
        
        #number of actions an agent chooses:
        self.my_act_dim = env.get_agent_act_dim()
        # same, but for all agents total:
        self.all_act_dim = self.my_act_dim * self.num_agents
        
        self.my_state_dim = env.get_agent_state_dim()
        self.all_state_dim = self.my_state_dim * self.num_agents
        
        #to remember unnoised acts for later policy evaluation
        self.unnoised_act_list = []
        
        learning_rate = param_dict["learning_rate"]
        
        if actor == "DDPG":
            #DDPG nets
            self.policy_net = PolicyNet(self.my_state_dim, self.my_act_dim).to(device)
            self.q_net = QNet(self.my_state_dim, self.my_act_dim ).to(device)

        elif actor == "MADDPG":
            #MADDPG nets
            self.policy_net = PolicyNet(self.my_state_dim , self.my_act_dim).to(device)
            self.q_net = QNet(self.all_state_dim, self.all_act_dim).to(device)
            
        self.critic_optimizer = optim.AdamW(self.q_net.parameters(), lr=learning_rate, amsgrad=True)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
            
        if self.has_supervisor:
            #pi : a_others --> a_mine
            #if env is deterministic (i.e. no hidden info), supervisor should approximate
            # the best response to knowing all others' actions.
            self.supervisor_policy_net = PolicyNet(self.my_state_dim + self.my_act_dim*(self.num_agents-1), self.my_act_dim).to(device)
            self.supervisor_policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
            
            # supervisor uses an MADDPG network so we have to train one if ddpg
            self.supervisor_q_net = QNet(self.my_state_dim, self.all_act_dim).to(device)
            self.supervisor_critic_optimizer = optim.AdamW(self.supervisor_q_net.parameters(), lr=learning_rate, amsgrad=True)
                
        self.discount_factor = param_dict["discount_factor"]
        
        self.critic_loss_fn = nn.MSELoss()
        
        memory_capacity = param_dict["memory_capacity"]
        self.memory = ReplayMemory(memory_capacity)
        
        #self.explore_type = param_dict["explore_type"]
        self.explore_params = param_dict["explore_params"] # should be a dictionary itself
        
        
        self.agent_id = agent_id
                
        self.critic_loss_history = []
        self.policy_loss_history = []
            
        if has_supervisor:
            self.supervisor_policy_loss_history = []
            self.supervisor_critic_loss_history = []
    
    def act(self, state, noise=True, remember_unnoised_act=True, act=None):
        if act is None:
            act = self.policy_net(state)
        else:
            pass #else use the act we were given
        
        try:
            assert act.shape == torch.Size([1,2])
        except:
            pass
        
        if self.BERTRAND:
            act[:,0] = 1.0 #Q = Q_max
        
            
        if remember_unnoised_act:
            self.unnoised_act_list.append(act)

        if noise:
            #state_noise = self.sample_state_noise()
            act_noise = self.sample_act_noise()
            act = act + act_noise
        
        if self.BERTRAND:
            act[:,0] = 1.0 #Q = Q_max
        
        
        
        #q must be >= 0
        #qmax = self.env.agent_q_maxes[self.agent_id]
        torch.abs_(act)
        
        """
        #set twice due to noise
        if self.COURNOT:
            act[0] = 1.0 #Q = Q_max
        """
        
        return act
    """        
    def sample_state_noise(self):
        mu_state_noise = self.explore_params["mu_state_noise"]
        std_state_noise = self.explore_params["std_state_noise"]
        
        state_noise = np.random.normal(loc=mu_state_noise, scale=std_state_noise, size=self.my_state_dim) #my_state_dim since we only want to noise it for the policy!
        state_noise = torch.from_numpy(state_noise).to(device=device,dtype=torch.float32)
        return state_noise
    """    
    def sample_act_noise(self):
        mu_act_noise = self.explore_params["mu_act_noise"]
        std_act_noise = self.explore_params["std_act_noise"]
        
        size = self.my_act_dim
        act_noise = np.random.normal(loc=mu_act_noise, scale=std_act_noise,size=size)
        act_noise = torch.from_numpy(act_noise).to(device=device,dtype=torch.float32)
        return act_noise
        
    
    def learn_from_batch(self, batch):
        #breakpoint()
        
        #MADDPG Learner uses states_n and acts_n
        #breakpoint()
        states = torch.zeros(
            size=[len(batch),self.num_agents,self.my_state_dim],
            dtype=torch.float32
            ).to(device)
        my_states = torch.zeros(
            size=[len(batch),self.my_state_dim],
            dtype=torch.float32
            ).to(device)
        acts = torch.zeros(
            size=[len(batch),self.num_agents,self.my_act_dim],
            dtype=torch.float32
            ).to(device)
        rewards = torch.zeros(
            size=[len(batch)],
            dtype=torch.float32
            ).to(device)
        
        for lv_tran, tran in enumerate(batch):
            states[lv_tran] = tran.state
            acts[lv_tran] = tran.act
            rewards[lv_tran] = tran.reward
        

        # batch is a batch of transitions sampled randomly
        other_idxs = [lv for lv in range(self.num_agents) if not lv == self.agent_id]
        
        #breakpoint()
        my_acts = acts[:,self.agent_id].flatten(start_dim=1)
        other_acts = acts[:,other_idxs].flatten(start_dim=1)#all except agent_id
        my_states = states[:,self.agent_id].flatten(start_dim=1)
        
        acts = acts.flatten(start_dim=1)
        
        target_q = rewards.unsqueeze(1) * self.env.param_dict["reward_scale"]
        
        # update q_net
        self.critic_optimizer.zero_grad()
        #breakpoint()
        if self.actor=="MADDPG":
            current_q = self.q_net(states.flatten(start_dim=1), my_acts, other_acts)
        elif self.actor=="DDPG":
            current_q = self.q_net(my_states.flatten(start_dim=1), my_acts)    
        critic_loss_val = self.critic_loss_fn(current_q, target_q)
        critic_loss_val.backward()
        self.critic_optimizer.step()
        self.critic_loss_history.append(critic_loss_val.item())
        
        #update policy
        self.policy_optimizer.zero_grad()
        if self.actor=="MADDPG":
            policy_acts = self.act(my_states, noise=False,remember_unnoised_act=False)
            policy_loss_val = -1*torch.mean(self.q_net(states.flatten(start_dim=1), policy_acts, other_acts))
        elif self.actor=="DDPG":
            policy_acts = self.act(my_states, noise=False,remember_unnoised_act=False)  
            policy_loss_val = -1*torch.mean(self.q_net(my_states.flatten(start_dim=1), policy_acts))
            
        policy_loss_val.backward()
        self.policy_optimizer.step()
        self.policy_loss_history.append(policy_loss_val.item())
        
#        breakpoint()
        if self.has_supervisor:
            # update q_supervisor (since it's a separate network)
            self.supervisor_critic_optimizer.zero_grad()
            supervisor_current_q = self.supervisor_q_net(my_states, my_acts, other_acts) #Q(s_i, a_i, a_-i)
            supervisor_critic_loss_val = self.critic_loss_fn(supervisor_current_q, target_q)
            supervisor_critic_loss_val.backward()
            self.supervisor_critic_optimizer.step()
            self.supervisor_critic_loss_history.append(supervisor_critic_loss_val.item())
            
            #update supervisor_policy
            self.supervisor_policy_optimizer.zero_grad()
            tmp = torch.cat((my_states, other_acts),dim=1) #(s_i, a_-i)
            supervisor_policy_acts = self.act(my_states, noise=False, remember_unnoised_act=False, act=self.supervisor_policy_net(tmp)) #pi_sup(s_i, a_-i)
            supervisor_policy_loss_val = -1*torch.mean(self.supervisor_q_net(my_states, supervisor_policy_acts, other_acts)) #-EQ(s_i, a_i=pi_sup(s_i,a_-i), a_-i)
            supervisor_policy_loss_val.backward()
            self.supervisor_policy_optimizer.step()
            self.supervisor_policy_loss_history.append(supervisor_policy_loss_val.item())
            
    
    def estimate_deficit(self, my_states, my_unnoised_acts, other_acts):
        assert self.has_supervisor
        #assumes that there are no state variables
        #deficit = MaxVal() - Expected val of state
        
        with torch.no_grad():
            #breakpoint()
            tmp = torch.cat([my_states,other_acts],dim=1)
            supervisor_acts = self.act(my_states, noise=False, remember_unnoised_act=False, act=self.supervisor_policy_net(tmp))
            optimal_q = self.supervisor_q_net(my_states, supervisor_acts, other_acts)
            actual_q = self.supervisor_q_net(my_states, my_unnoised_acts, other_acts)
            
            deficit = optimal_q - actual_q
            deficit = deficit / self.env.param_dict["reward_scale"]
        return deficit
        
    def calc_deficit_histories(self):
        #assert self.actor == "DDPG"
        other_idxs = [lv for lv in range(self.num_agents) if not lv == self.agent_id]
        self.deficit_history = []
        my_unnoised_act_arr = []
        other_acts_arr = []
        my_states = []
        #breakpoint()
        for tran, my_unnoised_act in zip(self.memory.full_memory, self.unnoised_act_list):
            state = tran.state
            my_state = state[self.agent_id,:].unsqueeze(0)
            acts = tran.act
            other_acts = acts[other_idxs,:].flatten(start_dim=1)
            
            my_unnoised_act_arr.append(my_unnoised_act)
            other_acts_arr.append(other_acts)
            my_states.append(my_state)
        
        #breakpoint()
        my_unnoised_acts = torch.cat(my_unnoised_act_arr)
        other_acts = torch.stack(other_acts_arr,dim=0).flatten(start_dim=1)
        my_states = torch.cat(my_states)
        
        self.deficit_history = self.estimate_deficit(my_states, my_unnoised_acts, other_acts)
        self.deficit_history = self.deficit_history.squeeze()
        if self.COURNOT:
            cournot_best_response_reward_history = self.cournot_best_response_profit(other_acts)
            self.cournot_deficit_history = cournot_best_response_reward_history - self.reward_history
        if self.BERTRAND:
            bertrand_best_response_reward_history = self.bertrand_best_response_profit(other_acts).to(device)
            self.bertrand_deficit_history = bertrand_best_response_reward_history - self.reward_history
            
    
    def calc_reward_history(self):
        #breakpoint()
        reward_history = []
        
        for tran in self.memory.full_memory:
            r = tran.reward
            reward_history.append(r)
        
        reward_history = torch.cat(reward_history)
        self.reward_history = reward_history
    
    def cournot_best_response_profit(self, other_acts):
        
        m = self.env.demand_slope_unsigned
        b = self.env.demand_intercept
        c = self.env.agent_MCs[self.agent_id]
        if self.env.num_agents == 1:
            q_others = torch.zeros(size=[len(self.reward_history)])
        else:
            #print(other_acts[0,:])
            self.env._scale_up_bid(other_acts)
            #print(other_acts[0,:])
            q_others = other_acts[:,0] #select Q's only
            #breakpoint()
        
        #TODO: I'm not sure if these are correct assuming Cournot
        # and I'm not sure how we should apply Cournot as the "ideal" here
        
        q_BR = (b-c)/(2*m) - 0.5*q_others
        p_BR = 1.5*(b-m*q_others) - 0.5*c
        cournot_best_response_reward_history = (p_BR-c)*q_BR 
        
        return cournot_best_response_reward_history 
    
    def bertrand_best_response_profit(self, other_acts):
        
        m = self.env.demand_slope_unsigned
        b = self.env.demand_intercept
        c = self.env.agent_MCs[self.agent_id]
        
        
        if self.env.num_agents == 1:
            p_others = torch.zeros(size=[len(self.reward_history)])
        else:
            #print(other_acts[0,:])
            assert abs(other_acts.shape[1]%2) < 1e-5 #there must be an even number of actions for (q,p) to be paired
            p_others = []
            for lv in range(self.env.num_agents - 1):
                self.env._scale_up_bid(other_acts[:,2*lv:2*(lv+1)])
                p_others.append(other_acts[:,2*lv+1]) # append p's
            #print(other_acts[0,:])
            p_others = torch.stack(p_others) #select P's only
        
        p_monopoly = (b + c)/2
        p_rival = torch.min(p_others,axis=0).values #TODO check axes for min
        
        if self.env.num_agents == 1:
            p_BR = p_monopoly
            q_BR = (b-p_BR)/m
            profit_BR = (p_BR-c)*q_BR
            bertrand_best_response_reward_history = [profit_BR] * len(self.reward_history)
            return torch.tensor(bertrand_best_response_reward_history)
        
        p_monopoly = torch.ones(size=p_rival.shape,device=device) * p_monopoly
        c = torch.ones(size=p_rival.shape,device=device) * c
        
        
        p_BR = torch.maximum(torch.minimum(p_rival, p_monopoly),c) #TODO check
        q_BR = (b-p_BR)/m #pd = b - m*q
        
        
        bertrand_best_response_reward_history = (p_BR - c)* q_BR
        #breakpoint()
        return bertrand_best_response_reward_history.to(device) 