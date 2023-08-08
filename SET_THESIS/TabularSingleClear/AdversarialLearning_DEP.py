#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:31:55 2023

@author: crw
"""
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

logging.basicConfig(level=logging.ERROR)
pypsa.opf.logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Get CPU or GPU device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

class MyModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 300, dtype=torch.float32),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 300, dtype=torch.float32),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, output_dim, dtype=torch.float32),
            #nn.tanh() #scale back out from [-1,1]
        )
        
    def forward(self, x):
        logits = self.network(x)
        return logits
        

my_loss_fn = nn.CrossEntropyLoss()

firm_state_dim=1
firm_act_dim=2
num_firms = 3
market_state_dim = 1
market_act_dim = 1
MADDPG_dim = num_firms * firm_act_dim + market_act_dim #assumes state dim don't count

firm_learner_arr = []
for lv in range(num_firms):
    actor = MyModel(input_dim=firm_state_dim, output_dim=firm_act_dim)
    critic = MyModel(input_dim=MADDPG_dim, output_dim=1)
    actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-4)
    
    learner = {"actor":actor, "critic":critic, "actor_optim":actor_optim, "critic_optim":critic_optim, "actor_loss_history":[], "critic_loss_history":[],}
    
    firm_learner_arr.append(learner)

market_actor = MyModel(input_dim=market_state_dim, output_dim=market_act_dim)
market_critic = MyModel(input_dim=MADDPG_dim, output_dim=1)
market_actor_optim = optim.Adam(market_actor.parameters(), lr=1e-4)
market_critic_optim = optim.Adam(market_critic.parameters(), lr=1e-4)
market_learner = {"actor":market_actor, "critic":market_critic, "actor_optim":market_actor_optim, "critic_optim":market_critic_optim, "actor_loss_history":[], "critic_loss_history":[],}

Q_SCALE = 200
P_SCALE = 40
R_SCALE = Q_SCALE*P_SCALE

def create_base_network(num_agents, clearing_window_hours):
    #num_snapshots = range(clearing_window_hours)
    num_snapshots = [1]
    base_network = pypsa.Network(snapshots = num_snapshots)
    
    W_per_MW = 1000
    
    #TODO set these in a smart way
    Q_max_MW = Q_SCALE
    MC = P_SCALE
    min_load_fraction = 0.0
    min_up_time_hours = 0 #3
    min_down_time_hours = 0 #3
    start_up_cost = 0 #5000
    shut_down_cost = 0 #5000
    ramp_limit_up_fraction = 1.0 #0.1
    ramp_limit_down_fraction = 1.0 #0.2 #TODO check that this shouldn't be like 1.0
    fixed_load_MW = 250
    
    VoLL_USD_per_MW = 100
    max_demand_MW = 300
    #unsigned_demand_slope_USD_per_MW = 2.0
    #demand_intercept_USD = 500
    #max_demand_MW = demand_intercept_USD / unsigned_demand_slope_USD_per_MW
    
    base_network.add("Bus", "bus")
    for lv_firm in range(num_agents):
        base_network.add(
            "Generator",
            f"Plant {lv_firm}",
            bus = "bus",
            #committable = True,
            #p_min_pu = min_load_fraction, #Q_min as %
            p_nom = Q_max_MW, #Q_max in MW
            p_min_pu = 0.0,
            p_max_pu = 1.0,
            #up_time_before = 0,
            #min_up_time = min_up_time_hours,
            #down_time_before = 0,
            #min_down_time = min_down_time_hours,
            #start_up_cost = start_up_cost,
            #shut_down_cost = shut_down_cost,
            #ramp_limit_up = ramp_limit_up_fraction,
            #ramp_limit_down = ramp_limit_down_fraction,
            marginal_cost = MC
            )
    #add consumer as a dummy generator with negative power gen
    base_network.add(
        "Generator",
        "Consumer Demand",
        bus="bus",
        p_max_pu = 0,
        p_min_pu = -1,
        p_nom = max_demand_MW,
        marginal_cost = VoLL_USD_per_MW,
    )
    base_network.add("Load", "load", bus="bus", p_set=[0])
    #breakpoint()    
    return base_network

from copy import deepcopy

def clear_market(market_act, firm_act_n):
    network = deepcopy(base_network)
    # set appropriate generator statuses
    p_cap = market_act.item() * P_SCALE
    network.generators.at["Consumer Demand", "marginal_cost"] = p_cap
    
    for lv_firm in range(num_firms):
        bid = firm_act_n[lv_firm]
        #bid[0] *= Q_SCALE no need to scale since its in pu
        bid[1] *= P_SCALE
        
        #p_bid = min(bid[1], p_cap) #market price cap
        network.generators.at[f"Plant {lv_firm}","p_max_pu"] = bid[0].item()
        network.generators.at[f"Plant {lv_firm}","marginal_cost"] = bid[1].item()
    
    #print("Begin opt")
    #breakpoint()
    network.optimize(display="/dev/null",log_fn="/home/crw/Programming/SET_THESIS/log.txt")
    #print("End opt")
    
    q_supply = network.generators_t.p.values.tolist()[0]
    q_sold = q_supply[-1] * (-1)
    q_demand = base_network.generators.p_nom["Consumer Demand"]
    if q_sold < q_demand:
        P_clear = base_network.generators.marginal_cost["Consumer Demand"]
    else:
        P_clear = network.buses_t.marginal_price["bus"][1]

    
    profits = (network.generators_t.p*(P_clear- base_network.generators.marginal_cost)).values.tolist()[0]
    firm_reward_n = profits[:-1]
    
    consumer_surplus = profits[-1]
    producer_surplus = sum(firm_reward_n)
    
    market_reward = alpha * consumer_surplus + (1 - alpha) * producer_surplus
    
    return firm_reward_n, market_reward

def train(num_episodes, batch_size):
    memory = {
        "firm_acts" : torch.zeros(size=[num_firms, firm_act_dim, num_episodes], dtype=torch.float32),
        "firm_rewards": torch.zeros(size=[num_firms, num_episodes], dtype=torch.float32),
        "market_acts" : torch.zeros(size=[1, market_act_dim, num_episodes], dtype=torch.float32),
        "market_rewards": torch.zeros(size=[1, num_episodes], dtype=torch.float32),
        }

    for lv_episode in range(num_episodes):
        #print("ep")
        with torch.no_grad():
            firm_obs_n = [ [0]*firm_state_dim] * num_firms
            firm_obs_n = torch.tensor(firm_obs_n, dtype=torch.float32)
            firm_act_n = [learner["actor"](firm_obs_n[lv]) for lv, learner in enumerate(firm_learner_arr)]
            firm_act_n = torch.stack(firm_act_n)
            torch.abs_(firm_act_n)
            
            memory["firm_acts"][:,:,lv_episode] = firm_act_n
            
            market_obs = [0]*market_state_dim
            market_obs = torch.tensor(market_obs, dtype=torch.float32)
            market_act = market_learner["actor"](market_obs)
            market_act.unsqueeze(0)
            torch.abs_(market_act)
            
            memory["market_acts"][:,:,lv_episode] = market_act
            
            firm_reward_n, market_reward = clear_market(market_act, firm_act_n)
            firm_reward_n = torch.tensor(firm_reward_n)
            market_reward = torch.tensor(market_reward)
            
            memory["firm_rewards"][:,lv_episode] = firm_reward_n
            memory["market_rewards"][:,lv_episode] = market_reward
        
        #now outside of no_grad:
        if lv_episode % batch_size == 0 and lv_episode > 0:
            print("Batch")
            for lv_learner in range(num_firms + 1):
                idxs = torch.randint(low=0, high=lv_episode, size=[batch_size]) #draw random idxs for each learner (different for each)
                firm_acts = memory["firm_acts"][:, :, idxs]
                blank_states = torch.zeros(size=[batch_size])
                market_acts = memory["market_acts"][:, :, idxs]
                
                if lv_learner < num_firms:
                    learner=firm_learner_arr[lv_learner]
                    Q_actual = memory["firm_rewards"][lv_learner, idxs]
                else:
                    learner = market_learner
                    Q_actual = memory["market_rewards"][0, idxs]
                
                all_acts = torch.cat([firm_acts.flatten(end_dim=1), market_acts.flatten(end_dim=1)], dim=1)
                
                #critic learn
                learner["critic_optim"].zero_grad()
                Q_pred = learner["critic"](all_acts)
                critic_loss_val = F.mse_loss(Q_pred, Q_actual)
                critic_loss_val.backward()
                learner["critic_optim"].step()
                learner["critic_loss_history"].append(critic_loss_val.item())
                
                #actor learn
                learner["actor_optim"].zero_grad()
                policy_acts = learner["actor"](blank_states) #stateless --> states all zero
                torch.abs_(policy_acts)
                all_acts[lv_learner,:,:] = policy_acts #estimate conditional value
                
                Q_policy = learner["critic"](all_acts)
                
                actor_loss_val = -1 * torch.mean(Q_policy)
                actor_loss_val.backward()
                learner["actor_optim"].step()
                learner["actor_loss_history"].append(actor_loss_val.item())
            
    return memory

batch_size = 128
base_network = create_base_network(num_firms, batch_size)
num_episodes = int(1e3)
alpha = 0.5

memory = train(num_episodes, batch_size)