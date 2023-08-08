#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:05:47 2023

@author: crw
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from collections import namedtuple
import torch
"""
AgentState = namedtuple(
    "Agentstate",
    ("demand_slope_unsigned", "demand_intercept", "Q_max", "MC")
    )
AgentAct = namedtuple(
    "AgentAct",
    ("Q_bid", "P_bid", "agent_id","cleared")
    )
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_slope_idx =0; state_intercept_idx = 1; state_Q_max_idx = 2; state_MC_idx = 3
act_Q_idx=0; act_P_idx=1; #act_Q_accepted_idx=2, act_id_idx=3; act_cleared_idx=4
bid_Q_idx=0; bid_P_idx=1; bid_Q_accepted_idx=2; bid_id_idx=3; bid_cleared_idx=4
bid_size=5
class ContinuousDAMarket():
    # a class implementing a physical grid, along with a redispatch market.
    # for now, t
    def __init__(self, param_dict, rng):
        self.param_dict = param_dict
        
        tmp = self.param_dict["state_distribution_dict"]["demand_slope_unsigned_distribution"]
        
        #for convenience, the number of agents
        self.num_firms = param_dict["num_learners"] + param_dict["num_naive"]
        self.num_agents = self.num_firms+1
        
        #initialize these arrays
        self.agent_Q_maxes = [None]*self.num_firms
        self.agent_MCs = [None]*self.num_firms
        
        self.alpha = self.param_dict["alpha"]
        
        #everyone uses same rng
        self.rng = rng
        
        self.reset()
    
    
    def get_agent_act_dim(self):
        return 2 #Q,P
        
    def get_agent_state_dim(self):
        return 4 #m, b, Q_max, MC
    
    def _set_demand_slope(self):
        #updates demand slope
        self.demand_slope_unsigned = self._random_attribute(attribute_name = "demand_slope_unsigned")

    def _set_demand_intercept(self):
        #updates demand intercept
        self.demand_intercept = self._random_attribute(attribute_name = "demand_intercept")
    
    def _random_attribute(self, attribute_name):
        #draws based on a specified distribution for a specific parameter
        
        #get the specified distribution (it is a lambda)
        lims = self.param_dict["state_distribution_dict"][attribute_name + "_distribution"]
        
        #draw from distribution
        ret = self.rng.uniform(low=lims[0],high=lims[1])
        #print(lims)
        
        return ret
    
    def _set_agent_states(self):
        #updates agent-seen variables Q_max and MC - only as stored in the env,
        #doesn't update agents themselves
                                                    
        for lv in range(self.num_firms):
            self.agent_Q_maxes[lv] = self._random_attribute(attribute_name = "Q_max")
            self.agent_MCs[lv] = self._random_attribute(attribute_name = "MC")
    
    def reset(self):
        #updates the environment configuration for another round
        self._set_demand_slope()
        self._set_demand_intercept()
        self._set_agent_states()
        pass
    
    def _scale_up_bid(self, bid):
        
        #NB THIS IS NOT THE OPPOSITE OF SCALE DOWN - it scales up actions like Q_bid, P_bid, which are NOT bounded to be in [0,1]
        Q_lims = self.param_dict["state_distribution_dict"]["Q_max" + "_distribution"]
        MC_lims = self.param_dict["state_distribution_dict"]["MC" + "_distribution"]
        
        Q_scale = (Q_lims[0] + Q_lims[1])/2.
        MC_scale = (MC_lims[0] + MC_lims[1])/2.
        if isinstance(bid, list):
            bid[0] *= Q_scale
            bid[1] *= MC_scale
        elif isinstance(bid,torch.Tensor):
            assert bid.shape[1] == 2
            bid[:,0] *= Q_scale
            bid[:,1] *= MC_scale
        else:
            assert False
    
    def _scale_down_attribute(self,val,attribute_name):
        lims = self.param_dict["state_distribution_dict"][attribute_name + "_distribution"]
        if abs(lims[1] - lims[0]) < 1e-4:
            #if there's no variation at all
            return 0
        else:
            val = (val - lims[0]) / (lims[1] - lims[0])
            return val
    
    def get_state_n(self):
        m = self.demand_slope_unsigned
        m_scaled = self._scale_down_attribute(m, "demand_slope_unsigned")
        b = self.demand_intercept
        b_scaled = self._scale_down_attribute(b, "demand_intercept")
        
        state_n = []
        for lv_agent in range(self.num_firms):
            Q_max = self.agent_Q_maxes[lv_agent]
            Q_max_scaled = self._scale_down_attribute(Q_max, "Q_max")
            
            MC = self.agent_MCs[lv_agent]
            MC_scaled = self._scale_down_attribute(MC, "MC")
            
            state = [m_scaled,b_scaled,Q_max_scaled,MC_scaled]
            state_n.append(state)
        
        #for market master:
        state = [0.0]*4
        state_n.append(state)
        
        return state_n
    
    def calculate_rewards(self, P_clear, bids_arr):
        #given the market clearing price and the (cleared) bid array
        #breakpoint()
        reward_n = torch.zeros(size=[len(bids_arr)+1])
        #print(bids_arr)
        for bid in bids_arr:
            agent_id = bid[bid_id_idx]
            #assert that bid has been trhough clearing process
            assert bid[bid_cleared_idx]
            
            per_Q_profit = P_clear - self.agent_MCs[agent_id]
            Q_max = self.agent_Q_maxes[agent_id]
            Q_accepted = bid[bid_Q_accepted_idx]
            
            if Q_accepted <= Q_max:
                 profit = per_Q_profit * Q_accepted
            else:
                 profit = per_Q_profit * Q_max - P_clear * (Q_accepted - Q_max)
            
            reward_n[agent_id] = profit
        Q_tot = sum([bid[bid_Q_accepted_idx] for bid in bids_arr])
        
        producer_surplus = torch.sum(reward_n).item()
        consumer_surplus = 0.5 * (self.demand_intercept - P_clear) * Q_tot
        reward_n[-1] = self.alpha*consumer_surplus + (1-self.alpha) * producer_surplus
        return reward_n
    
    def clear_market_calculate_profits_no_update(self, act_n, PLOT_BIDS=False):
        #clear market
        P_clear, bids_arr = self.clear_market(act_n)
        
        reward_n = self.calculate_rewards(P_clear, bids_arr)
        
        if PLOT_BIDS:
            #breakpoint()
            self.plot_bids(P_clear, bids_arr)
            
        return P_clear, reward_n
    
    def clear_market(self, act_n):
        #price cap:
        price_cap_act = act_n[-1]
        self._scale_up_bid(price_cap_act)
        P_cap = price_cap_act[1]
        
        b=self.demand_intercept
        m=self.demand_slope_unsigned
        
        Q_crossover =  (b-P_cap) / m
        
        bid_act_n = act_n[:-1]
        for lv_agent, act in enumerate(bid_act_n):
            act.append(lv_agent)
            
        #given a list of unsorted bids,act_n
        P_demand = lambda Q: min(P_cap, b - m * Q)
        
        #act_n_list = act_n.tolist()
        sorted_acts = sorted(bid_act_n, key = lambda x: x[act_P_idx]) #whacky way to do it but oke sure
        
        #we'll want to include more information in the 'bids' structure
        bids_arr = []
        #breakpoint()
        Q_accepted = 0
        for act in sorted_acts:
            bid = [act[0], act[1], 0, act[2], False]
            self._scale_up_bid(bid)
            #[Q,P,Q_accepted, ID, cleared?]
            
            #mark for verification that this bid has been cleared
            bid[bid_cleared_idx] = True
            
            P_next = bid[bid_P_idx]
            if P_demand(Q_accepted) > P_next:
                #if there's still WTP for more
                Q_next = bid[bid_Q_idx]
                P_clear = P_next
                if P_demand(Q_accepted + Q_next) > P_next:
                    #if taking this whole bid doesn't exhaust demand, accept it all
                    Q_accepted += Q_next
                    bid[bid_Q_accepted_idx]= Q_next
                else:
                    #else, this is the last bid, and we accept a fraction only
                    Q_tot = (b-P_next) / m
                    dQ = Q_tot - Q_accepted
                    
                    
                    #check the assumption that gets us to this control block
                    assert dQ >= 0 and dQ <= Q_next
                    
                    Q_accepted += dQ
                    #by assumption, the demand and supply curves intersect:
                    assert abs(P_demand(Q_accepted) - P_next) < 1e-3
                    
                    bid[bid_Q_accepted_idx] = dQ
                    
            else: #if there's no more WTP, set Q_accepted = 0
                bid[bid_Q_accepted_idx]= 0
                
                #if NO bids were processed, then P_clear will not be set
                #triggers this block, which then sets P_clear arbitrarily
                try:
                    P_clear
                except NameError:
                    #breakpoint()
                    P_clear = P_demand(0)
            
            bids_arr.append(bid)            
        #breakpoint()
        P_clear = max(P_clear, P_demand(Q_accepted))
        return P_clear, bids_arr
    
    def plot_bids(self,P_clear,bids_arr):
        m=self.demand_slope_unsigned
        b=self.demand_intercept
        x_max_demand = b/m
        hardcoded_janky_Q_max_per_agent = 100 #should be the max possible Q_max, want it to be same for all draws for plot scaling
        x_max_supply = self.param_dict["state_distribution_dict"]["Q_max" + "_distribution"][1]*self.num_agents * 1.1
        
        x_max = max([x_max_demand,x_max_supply])
        # DATA TO PLOT
        x_start = 0
        x_stop = x_max
        num_points =1000
        x = np.linspace(start=x_start, stop=x_stop, num=num_points) 
        y_demand = b - m * x
        y_demand_monopoly = b - 2*m*x
        
        # PLOTTING PARAMETERS
        x_min = -10
        #x_max = 
        y_min = -10
        y_max = b*2
        x_label = "Power (MW)"
        y_label = "Price (USD)"
        plot_title = "Supply and Demand"
        plot_subtitle = "SUBTITLE_NA"
        
        # PLOT COMMANDS
        plt.figure()
        plt.plot(x, y_demand)
        plt.plot(x,y_demand_monopoly)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([x_min,x_max])
        plt.ylim([y_min,y_max])
        plt.suptitle(plot_title)
        plt.title(plot_subtitle)
        
        plt.hlines(y=P_clear, xmin=x_min, xmax=x_max,linestyles="--")

        #breakpoint()
        #Q_total = 0
        x_supply_bid = [0]
        y_supply_bid = [0]
        x_supply_real = [0]
        y_supply_real = [0]
        for lv_bid, bid in enumerate(bids_arr):
            #breakpoint()
            Q_bid = bid[bid_Q_idx]
            P_bid = bid[bid_P_idx]
            #up the staircase
            x_supply_bid.append(x_supply_bid[-1])
            y_supply_bid.append(P_bid)
            #and then over
            x_supply_bid.append(x_supply_bid[-1] + Q_bid)
            y_supply_bid.append(P_bid)
        
        #same staircase plot, but now for the "real" supply curve
        mc_arr = [[self.agent_Q_maxes[lv], self.agent_MCs[lv]] for lv in range(self.num_agents)]
        mc_bids_arr = sorted(mc_arr, key=lambda x: x[1])
        for lv_bid, bid in enumerate(mc_bids_arr):
            Q_bid = bid[bid_Q_idx]
            P_bid = bid[bid_P_idx]
            #up the staircase
            x_supply_real.append(x_supply_real[-1])
            y_supply_real.append(P_bid)
            #and then over
            x_supply_real.append(x_supply_real[-1] + Q_bid)
            y_supply_real.append(P_bid)        

        plt.plot(x_supply_bid,y_supply_bid,color="orange")
        plt.plot(x_supply_real,y_supply_real,color="orange",linestyle="--")
        plt.legend(["Demand","Monopoly Demand","P_clear","Bids","Actual Supply"])
        plt.show()
        #breakpoint()