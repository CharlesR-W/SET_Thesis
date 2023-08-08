#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:49:31 2023

@author: crw
"""

import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

class DayAheadMarketEnvironment(gym.Env):
    
    def __init__(self, param_dict):
        self.param_dict=param_dict
        self.Q_MAX_MW = param_dict["Q_MAX_MW"]
        self.P_MAX_USD = param_dict["P_MAX_USD"]
        
        self.DISCRETIZED = param_dict["DISCRETIZED"]
        #determines if the action and observation spaces are discretized - for use with tabular learning
        
        self.num_agents = param_dict["num_agents"]
        #the number of agents which will be competing in the environment
        
        self.agent_bids_allowed = param_dict["agent_bids_allowed"]
        #the number of (Q,P) pairs a single agent is allowed to submit
        
        self.current_round = 0
        self.train_rounds = param_dict["train_rounds"]
        # maximum number of rounds per episode
        
        if self.DISCRETIZED:
            
            ####################### state / observation variables
            #generator Q_max
            self.num_allowed_Q_maxes = param_dict["num_allowed_Q_maxes"]
            self.allowed_Q_maxes_MW = np.linspace(start=100,stop=300, num=self.num_allowed_Q_maxes)
            self.allowed_Q_maxes_MW = list(self.allowed_Q_maxes_MW)
            
            #generator MCs
            self.num_allowed_MCs = param_dict["num_allowed_MCs"]
            self.allowed_MCs_USDperMW = np.linspace(start=50,stop=100, num=self.num_allowed_MCs)
            self.allowed_MCs_USDperMW = list(self.allowed_MCs_USDperMW)
            
            #demand-slopes
            self.num_allowed_demand_slopes = param_dict["num_allowed_demand_slopes"]
            self.allowed_demand_slopes_USDperMW = np.linspace(start=1,stop=10, num=self.num_allowed_demand_slopes)
            self.allowed_demand_slopes_USDperMW = list(self.allowed_demand_slopes_USDperMW)
            
            #demand-slope intercepts
            self.num_allowed_demand_intercepts = param_dict["num_allowed_demand_intercepts"]
            self.allowed_demand_intercepts_USD = np.linspace(start=100,stop=300, num=self.num_allowed_demand_intercepts)
            self.allowed_demand_intercepts_USD = list(self.allowed_demand_intercepts_USD)
            
            ####################### action-spaces
            #ratios of Q which agent can bid in (of their own Q)
            self.num_allowed_Q_ratios = param_dict["num_allowed_Q_ratios"]       
            self.allowed_Q_ratios = np.linspace(start=0.5,stop=1.0, num=self.num_allowed_Q_ratios)    
            
            #ratios of MC which agents can bid in (their own MC)
            self.num_allowed_P_ratios = param_dict["num_allowed_P_ratios"]
            self.allowed_P_ratios = np.linspace(start=0.8,stop=1.2, num=self.num_allowed_P_ratios)
            self.allowed_P_ratios = list(self.allowed_P_ratios)
            

        self.action_space = self.get_multiagent_action_space()
        #assign action_space
        
        self.observation_space = self.get_multiagent_obs_space()
        #assign observation_space
        #if you wanted, could return the same obs to all agents - for now leave with multiple copies since more general.
        
        #min and max rewards
        reward_max = self.Q_MAX_MW * self.P_MAX_USD
        reward_min = - reward_max
        
        self.reward_range = [reward_min, reward_max]
        
        #P_demand(Q) [USD] = intercept [USD] - (unsigned_slope [USD/MW] ) * ( Q [MW] ) 
        self.demand_slope_unsigned_USDperMW = None
        self.demand_intercept_USD = None
        
        self.set_agent_MCs()
        self.set_agent_Q_maxes()
        self.set_demand_function()
        #sets the slope and intercept
        '''        
    def agent_allowed_actions(self):
        assert self.DISCRETIZED
        #can only list allowed actions if they are discrete
        '''        
    def get_multiagent_action_space(self):
        # returns a space object for Gym, formatted as:
        # [agent1 [(QP),(QP),...], agent 2 [(QP),(QP),...], ...]
        #(the square brackets there are for clarity - all must be tuples still)
        
        action_space =[]
        for n in range(self.num_agents):
            agent_space = []            
            for b in range(self.agent_bids_allowed):
                if self.DISCRETIZED:
                    #For use if discretising Q,P space
                    Q_space = gym.spaces.Discrete(self.num_allowed_Q_ratios)
                    P_space = gym.spaces.Discrete(self.num_allowed_P_ratios)
                else:
                    #if using a continuous Q and P space, first find the ranges for each
                    Q_min_MW = 0
                    Q_max_MW = self.QMAX_MW / self.agent_bids_allowed
                    # divide to make sure agent can bid Q_max in total
                    
                    P_min_USD = -self.PMAX_USD
                    P_max_USD = self.PMAX_USD
                    
                    Q_space = gym.spaces.Box(Q_min_MW, Q_max_MW)
                    P_space = gym.spaces.Box(P_min_USD, P_max_USD)
                
                bid_space = gym.spaces.Tuple([Q_space, P_space])
                agent_space.append(bid_space)
                #array of tuples of spaces [T(ab), T(ab)...]
                
            agent_space = gym.spaces.Tuple(agent_space)
            action_space.append(agent_space)
            
        ret = gym.spaces.Tuple(action_space)
        #tuple of shape: n_agents * (num_Qs * num_Ps)
        return ret
    
    def get_multiagent_obs_space(self):
        #returns the structured 'gym.space' object corresponding to num_agents copies of the individual agent observation spaces
        if self.DISCRETIZED:
            ret = []
            for lv in range(self.num_agents):
            #for each agent, create the relevant space-objects and bunch them together
                Q_max_space = gym.spaces.Discrete(self.num_allowed_Q_maxes)
                MC_space = gym.spaces.Discrete(self.num_allowed_MCs)
                demand_slope_space = gym.spaces.Discrete(self.num_allowed_demand_slopes)
                demand_intercept_space = gym.spaces.Discrete(self.num_allowed_demand_intercepts)
                
                agent_space = [Q_max_space, MC_space, demand_slope_space, demand_intercept_space]
                agent_space = gym.spaces.Tuple(agent_space)
                #composite space
                
                ret.append(agent_space)
                
            ret = gym.spaces.Tuple(ret)
        else:
        #if continuous
            raise NotImplementedError
            
        return ret
    
    
    def set_demand_function(self):
        if self.DISCRETIZED:
            self.demand_slope_unsigned_USDperMW = random.choice(self.allowed_demand_slopes_USDperMW)
            self.demand_intercept_USD = random.choice(self.allowed_demand_intercepts_USD)
        else:
            raise NotImplementedError
            
    def step(self, act_n):
        
        self.set_agent_MCs()
        self.set_agent_Q_maxes()
        self.set_demand_function()
        
        #calculate the outcome of this round, and assign rewards
        reward_n = self.clear_market(act_n)
        
        #move to next timestep
        self.current_round += 1
        
        self.set_demand_function()
        observation_n = self.get_observation_n()
        
        #should we be finished?
        if self.current_round >= self.train_rounds:
            done_n = [True]*self.num_agents
        else:
            done_n = [False]*self.num_agents
            
        info_n = None #I don't know what this is really
        return act_n, observation_n, reward_n, done_n, info_n
    
    def set_agent_Q_maxes(self):
        #assigns the total generation capacity of each agent as a tuple (Qmax1,Qmax2,...)
        tmp = []
        for n in range(self.num_agents):
            r = random.choice(self.allowed_Q_maxes_MW)
            tmp.append(r)
            
        self.agent_Q_maxes_MW = tuple(tmp)
        
    def set_agent_MCs(self):
        #sets the marginal cost of each agent's generation - as part of the randomisation.  If no randomisation is desired, simply make the list of length 0.
        if self.DISCRETIZED:
        #MCs allowed by the discretisation
            self.agent_MCs_USDperMW = [random.choice(self.allowed_MCs_USDperMW) for n in range(self.num_agents)]
            #select n times from that MC set
        else:
            raise NotImplementedError
        
    def get_observation_n(self):
        #returns a tuple of observations - one for each agent
        
        observation_n = []
        for n in range(self.num_agents):
            Q_max = self.agent_Q_maxes_MW[n]
            Q_max_idx = self.allowed_Q_maxes_MW.index(Q_max)
            
            MC = self.agent_MCs_USDperMW[n]
            MC_idx = self.allowed_MCs_USDperMW.index(MC)
            
            m = self.demand_slope_unsigned_USDperMW
            m_idx = self.allowed_demand_slopes_USDperMW.index(m)
            
            b = self.demand_intercept_USD
            b_idx = self.allowed_demand_intercepts_USD.index(b)
            
            obs = (Q_max_idx,MC_idx,m_idx,b_idx)
            #combine into tuple
            
            observation_n.append(obs)
            #append to array
            
        ret = tuple(observation_n)
        #make into tuple
        
        return ret
    
    def clear_market(self,act_n):
        Q_idx=0
        P_idx=1
        N_idx=2
        tmp = [[ (QP[Q_idx], QP[P_idx], lv_agent) for QP in act] for lv_agent,act in enumerate(act_n)]
        
        bids_unsorted = []
        for agent_bids in tmp:
            for bid in agent_bids:
                Q_ratio = self.allowed_Q_ratios[bid[Q_idx]]
                P_ratio = self.allowed_P_ratios[bid[P_idx]]
                agent_idx = bid[N_idx]
                Q_max = self.agent_Q_maxes_MW[agent_idx]
                MC = self.agent_MCs_USDperMW[agent_idx]
                Q = Q_max * Q_ratio
                P = MC * P_ratio
                bid = (Q,P,agent_idx)
                bids_unsorted.append(bid)
        #add the agent's number to the bid so we can track it, the * unpacks these
        
        bids_sorted = sorted(bids_unsorted, key=lambda QPN: QPN[P_idx])
        
        
        Q_supply = 0
        P_supply = 0
        P_demand = lambda Q: self.demand_intercept_USD - self.demand_slope_unsigned_USDperMW * Q
        lv_bid = 0
        bid_Q_idx = 0
        bid_P_idx = 1
        done = False
        while not done:
            next_bid = bids_sorted[lv_bid]
            #get the bid with the next-highest price
            Q_next = next_bid[bid_Q_idx]
            Q_tmp = Q_supply + Q_next
            P_supply = next_bid[bid_P_idx]
            if P_demand(Q_tmp) < P_supply:
                # WTP < P_supply, we have to split the interval
                Q_final = -(P_supply - P_demand(Q_supply)) / self.demand_slope_unsigned_USDperMW
                delta_Q_final = Q_final - Q_supply
                #P0 - m(QS + Qnext) = PS --> Qnext = (P0 - PS)/m - QS (but P0-mQS is the lambda function so its easy to just use that)
                
                assert delta_Q_final <= Q_next
                #assert delta_Q_final >= 0
                #confirm the assumption holds, as we use Q_final later
                
                P_clear = P_supply
                #the clearing price
                
                done = True
                lv_last_bid = lv_bid
                #keep track of the last bid to be accpted
                break
            else:
                #otherwise we accept the whole bid
                Q_accepted = Q_next
                Q_supply += Q_accepted
                done = P_demand(Q_supply) < P_supply
                assert not done or lv_bid == len(bids_sorted)-1
            
            lv_bid += 1
            if lv_bid >= len(bids_sorted):
                #shortage conditions - if there is not enough Q to meet demand
                P_clear = P_demand(Q_tmp)
                done = True
                
                lv_last_bid = lv_bid
                #track the last bid we accept
        
        #self.plot_bids(bids_sorted,P_clear)
        assert P_clear >= 0
        
        
        #now calculate all the rewards for each agent
        reward_n = [0]*self.num_agents
        Q_idx = 0
        agent_idx = 2
        for lv_bid, QPN in enumerate(bids_sorted):
            if lv_bid == lv_last_bid:
                Q_accepted = delta_Q_final
                #this is the block where we don't necessarily accept it all
            elif lv_bid >= lv_last_bid:
                #we can ignore the remaining bids
                break
            else:
                Q_accepted = QPN[Q_idx]
    
            #income = P_clear * Q_accepted
            agent_id = QPN[agent_idx]
            profit = (P_clear - self.agent_MCs_USDperMW[agent_id]) * Q_accepted
            reward_n[agent_id] += profit
        
        #print(f"MCs: {self.agent_MCs_USDperMW}")
        #print("Q_maxes: {self.agent_Q_maxes_MW}")
        #print(f"bids_sorted: {bids_sorted}")
        #print(f"P_clear: {P_clear}")
        #print(f"reward_n: {reward_n}")
        return reward_n
    
    def plot_bids(self, bids_sorted,P_clear):
        #bid format is QPN
        x = []
        y = []
        Q_idx = 0
        P_idx = 1
        agent_idx = 2
        Q = 0
        x.append(0)
        y.append(0)
        for QPN in bids_sorted:
            x.append(Q)
            P = QPN[P_idx]
            y.append(P)
            
            Q += QPN[Q_idx]
            
            x.append(Q)
            y.append(P)
        fig = plt.figure()
        plt.plot(x,y)
        
        x_max = self.demand_intercept_USD / self.demand_slope_unsigned_USDperMW
        #P0 -mx =0 --> x = P0/m
        num_pts = 100
        x_demand = np.linspace(start=0,stop=x_max,num=num_pts)
        y_demand = self.demand_intercept_USD - self.demand_slope_unsigned_USDperMW * x_demand
        plt.plot(x_demand,y_demand)
        plt.xlabel("Quantity MW")
        plt.ylabel("Price/MW USD")
        plt.legend(["Supply", "Demand"])
        plt.axhline(P_clear)
        plt.show()
        return
    
    def set_agents(self, agents_arr):
        self.agents_arr = agents_arr
    
    def reset(self):
        raise NotImplementedError
        
    def render(self,):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
        
    def calculate_per_agent_monopoly_surplus(self):
        #clears the market as if agents were plants controlled by a single agent
        #NB does NOT respect the tabular action-constraints, does respect MCs and Q_maxes
        
        
    def calculate_per_agent_bertrand_surplus(self):