#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:05:47 2023

@author: crw
"""
import numpy as np

class PhysicalGridMarket():
    # a class implementing a physical grid, along with a redispatch market.
    def __init__(self, test_network, param_dict):
        self.network = test_network
        self.param_dict = param_dict
        
    class AgentObs():
        def __init__(self, demand_slope_unsigned, demand_intercept, Q_max, MC):
            self.demand_slope_unsigned = demand_slope_unsigned
            self.demand_intercept = demand_intercept
            self.Q_max = Q_max
            self.MC = MC
        
    class AgentAct():
        
        def __init__(self,Q_bid, P_bid, agent_id):
            self.Q_bid = Q_bid
            self.P_bid = P_bid
            self.agent_id = agent_id #??? how to get wisely?
            self.Q_accpted = 0
            
            #tracks whether this bad has been through a market-clearing or not,
            # for verification purposes
            self.cleared = False
    
    def _set_demand_slope(self):
        #updates demand slope
        self.demand_intercept = self._random_attribute(attribute_name = "demand_slope")

    def _set_demand_intercept(self):
        #updates demand intercept
        self.demand_intercept = self._random_attribute(attribute_name = "demand_intercept")
    
    def _random_attribute(self, attribute_name):
        #draws based on a specified distribution for a specific parameter
        
        #get the specified distribution (it is a lambda)
        distn = self.param_dict[attribute_name + "_distribution"]
        
        #draw from distribution
        tmp = distn()
        
        return tmp
    
    def _set_agent_states(self):
        #updates agent-seen variables Q_max and MC
        for lv in range(self.num_agents):
            self.agent_Q_maxes[lv] = self.draw_Q_max()
            self.agent_MCs[lv] = self.draw_MC()
        
    
    def next_configuration(self):
        #updates the environment configuration
        self._set_demand_slope()
        self._set_demand_intercept()
        self._set_agent_states()
        
    
    def get_agent_obs_n(self):
        m = self.demand_slope_unsigned
        b = self.demand_intercept
        
        obs_n = []
        for lv_agent in range(self.num_agents):
            Q_max = self.agent_Q_maxes[lv_agent]
            MC = self.agent_MCs[lv_agent]
            
            obs = self.AgentObs(demand_slope_unsigned=m, demand_intercept=b, Q_max=Q_max, MC=MC)
            obs_n.append(obs)
        
        return obs_n
    
    """
    I'm going to split STEP into different functions
    
    def step(self):
        pass
    """
    def calculate_profits(self, P_clear, act_n):
        #given the market clearing price and the (cleared) bid array
        reward_n = []
        for lv_agent, act in enumerate(act_n):
            
            #assert that bid has been trhough clearing process
            assert act.cleared
            
            per_Q_profit = P_clear - self.agent_MCs[lv_agent]
            profit = per_Q_profit * act.Q_accepted
            
            reward_n.append(profit)
        
        return reward_n
    
    def clear_and_get_profits(self, act_n):
        #clear market
        P_clear = self.clear_market(act_n)
        
        reward_n = self.calculate_profits(P_clear, act_n)
        
        self.next_configuration()
        obs_n = self.get_agent_obs_n()
        
        #now update each agent's observation
        for lv_agent in range(self.num_agents):
            obs = obs_n[lv_agent]
            self.agents[lv_agent].set_new_obs(obs)
    
    def clear_market(self, act_n):
        #given a list of unsorted bids,act_n
        P_demand = lambda Q: self.demand_intercept - self.demand_slope_unsigned * Q
        
        sorted_bids = sorted(lambda act: act.P_bid, act_n)
        
        Q_accepted = 0
        for bid in sorted_bids:
            #mark for verification that this bid has been cleared
            bid.cleared = True
            P_next = bid.P_bid
            if P_demand(Q_accepted) < P_next:
                #if there's still WTP for more
                Q_next = bid.Q_bid
                P_clear = P_next
                if P_demand(Q_accepted + Q_next) < P_next:
                    #if taking this whole bid doesn't exhaust demand, accept it all
                    Q_accepted += Q_next
                    bid.Q_accepted = bid.Q_bid
                else:
                    #else, this is the last bid, and we accept a fraction only
                    dQ = -1*( P_next - P_demand(Q_accepted) ) / self.demand_slope_unsigned
                    
                    #check the assumption that gets us to this control block
                    assert dQ >= 0 and dQ <= Q_next
                    
                    Q_accepted += Q_next
                    #by assumption, the demand and supply curves intersect:
                    assert abs(P_demand(Q_accepted) - P_next) < 1e-3
                    
                    bid.Q_accepted = dQ
                    
            else: #if there's no more WTP, set Q_accepted = 0
                bid.Q_accepted = 0
                
        #returns P_clear
        return P_clear