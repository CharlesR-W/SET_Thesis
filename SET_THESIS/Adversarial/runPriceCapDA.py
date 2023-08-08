#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:28:45 2023

@author: crw
"""
import numpy as np
from PriceCapDA import ContinuousDAMarket
from SingleBidMADDPGLearner import SingleBidLearner
""" [MOST] OF THE CODE BELOW IS COPIED AND LIGHTLY EDITED FROM THE PYTORCH TUTORIAL ON CARTPOLE RL
CREDIT TO THEM FOR THIS"""

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
def perform_run(env_param_dict, agent_param_dict, learn_interval, batch_size, num_episodes,  num_learners, random_seed, run_name, num_naive=0, is_MADDPG=True, PLOT_BIDS=True, has_supervisor=False,COURNOT=False,BERTRAND=False):
    
    rng = np.random.default_rng(random_seed)
    agent_param_dict["rng"] = rng
    
    number_noisy_episodes = int(0.5 * num_episodes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_agents = num_naive + num_learners + 1
    env = ContinuousDAMarket(param_dict=env_param_dict, rng=rng)
    learner_arr = [
        SingleBidLearner(
        param_dict=agent_param_dict,
        env=env,
        agent_id=lv,
        actor="MADDPG" if is_MADDPG else "DDPG",
        has_supervisor = has_supervisor,
        COURNOT = COURNOT,
        BERTRAND = BERTRAND,
        )
        for lv in range(num_learners)
        ]
    
    market_learner = SingleBidLearner(
    param_dict=agent_param_dict,
    env=env,
    agent_id=num_learners,
    actor="MADDPG" if is_MADDPG else "DDPG",
    has_supervisor = has_supervisor,
    COURNOT = COURNOT,
    BERTRAND = BERTRAND,
    IS_MASTER=True
    )
    
    reward_n_history = torch.zeros(size=[num_agents, num_episodes])
    P_clear_history = torch.zeros(size=[num_episodes])
    for lv_episode in range(num_episodes):
        with torch.no_grad():
            env.reset()
            state_n = env.get_state_n()
            state_n = torch.tensor(state_n, dtype=torch.float32, device=device).unsqueeze(0)
            #print(f"state_n shape: {state_n.shape}")
            act_is_noisy = True if lv_episode < number_noisy_episodes else False
            act_firms = [learner.act(state_n[:,lv_learner,:], noise=act_is_noisy, remember_unnoised_act=True) for lv_learner, learner in enumerate(learner_arr)] + [torch.tensor([[1,1]], device=device)] * num_naive
            
            act_market = market_learner.act(state_n[:,-1,:], noise=act_is_noisy) #ASSUMES STATELESS
            act_n = act_firms + [act_market]
            
            act_n_tensor = torch.stack(act_n).squeeze(1)
            act_n = act_n_tensor.tolist()
            
            
            
            if lv_episode % (int(num_episodes/5)) == 0:
                plot_bids=PLOT_BIDS
                #print(f"Episode: {lv_episode}")
            else:
                plot_bids=False
            P_clear, reward_n = env.clear_market_calculate_profits_no_update(act_n, plot_bids)
            
            P_clear_history[lv_episode] = P_clear
            
            assert P_clear <= act_n[-1][1]
            
            # assume that our environment is single-step
            done = True
            terminated = True 
            next_state_n = [None]*(num_learners+1)
            
            #breakpoint()
            reward_n = torch.tensor(reward_n, dtype=torch.float32, device=device)
            
            #store it for history
            reward_n_history[:,lv_episode] = reward_n
            
            reward_n.unsqueeze_(1)
    
            #push the transition to each learner
            for lv_learner, learner in enumerate(learner_arr):
                #breakpoint()
                learner.memory.push(state_n.squeeze(0), act_n_tensor, next_state_n, reward_n[lv_learner])
                
            market_learner.memory.push(state_n.squeeze(0), act_n_tensor, next_state_n, reward_n[-1])
        
        #NOW outside of no_grad!
        # Store the transition in memory, and then learn
        if lv_episode % learn_interval == 0 and lv_episode >= batch_size :
            #print(reward_n)
            for lv_learner, learner in enumerate(learner_arr):
                # Perform one step of the optimization (on the policy network)
                
                batch = learner.memory.sample(batch_size=batch_size)
                learner.learn_from_batch(batch=batch)
                
            batch = market_learner.memory.sample(batch_size=batch_size)
            market_learner.learn_from_batch(batch=batch)
    
    P_clear_history /= 40 #MC
    #breakpoint()
    return learner_arr, market_learner, P_clear_history
    
    print('Complete')