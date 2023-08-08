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
import itertools

class SingleBidContinuousDayAheadMarketEnvironment(gym.Env):
    
    def __init__(self, param_dict):
        self.param_dict=param_dict
        
        #agents append themselves when created
        self.agents = []

        #the number of agents which will be competing in the environment        
        self.num_agents = param_dict["num_agents"]

        #assign action_space
        self.action_space = self.get_multiagent_action_space()
        
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
        
        #one-time calculation of hidden-states, used in the expectation operations as part of the policy deficit estimation
        self.calculate_allowed_hidden_states()
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
        
        #calculate the outcome of this round, and assign rewards
        self.current_act_n = act_n
        reward_n = self.clear_market_as_bid()
        
        if self.PLOT_AT_COST_OUTCOME or self.CALCULATE_REGRET:
            self.calculate_at_cost_outcome()
            
        if self.PLOT_MONOPOLY_OUTCOME or self.CALCULATE_REGRET:
            self.calculate_monopoly_outcome()
            
        if self.CALCULATE_REGRET:
            regret_n = self.calculate_monopoly_regret()
        


        observation_n = self.get_observation_n()
        
        #should we be finished? #assumes we will update right after
        if self.current_round + 1 >= self.train_rounds:
            done_n = [True]*self.num_agents
        else:
            done_n = [False]*self.num_agents
            
        info_n = None #I don't know what this is really
        
        if self.CALCULATE_REGRET:
            return act_n, observation_n, reward_n, regret_n, done_n, info_n
        else:
            return act_n, observation_n, reward_n, done_n, info_n
    
    def update_new_round(self):
        self.set_agent_MCs()
        self.set_agent_Q_maxes()
        self.set_demand_function()
        self.set_demand_function()
        
        #move to next timestep
        self.current_round += 1
        
        obs_n = self.get_observation_n()
        
        #return the new observation
        return obs_n
    
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
        
        self.current_obs_n = ret
        
        return ret
    
    def calculate_profit(self, QPNQ_arr, P_clear, obs_n):
        reward_n = [0]*self.num_agents
        #bids_arr = []
        Q_bid_idx = 0
        Q_accepted_idx = 3
        agent_idx = 2
        MC_idx = 1
        for lv_bid, QPNQ in enumerate(QPNQ_arr):
            Q_accepted = QPNQ[Q_accepted_idx]
            agent_id = QPNQ[agent_idx]
            
            #get the MC
            MC_id = obs_n[agent_id][MC_idx]
            MC = self.allowed_MCs_USDperMW[MC_id]
            
            profit = (P_clear - MC) * Q_accepted
            reward_n[agent_id] += profit
        return reward_n
    
    def clear_market_as_bid(self):
        #in step(), current_act_n is set immediately before the call to this function
        act_n = self.current_act_n
        obs_n = self.current_obs_n
        QPNQ_arr, P_clear, reward_n = self.clear_market(act_n,obs_n)
        
        #now calculate all the rewards for each agent
        #reward_n = self.calculate_profit(QPNQ_arr, P_clear)
        
        self.current_reward_n = reward_n
        self.current_P_clear = P_clear
        self.current_bids_arr = QPNQ_arr
        return self.current_reward_n
    
    def clear_market(self, act_n, obs_n):
        #obs_n is the set of ASSERTED agent observations - one must use self.current_obs_n to get the 'real ones'
        #act_n = [(Q,P), (Q,P),...]
        Q_idx=0
        P_idx=1
        N_idx=2
        
        Q_max_idx = 0
        MC_idx = 1
        
        bids_unsorted = []
        #convert the bid from the integer table-addresses to QPN tuples
        for lv_agent, QP in enumerate(act_n):
            #get the agent's Q_max
            Q_max_id = obs_n[lv_agent][Q_idx]
            Q_max = self.allowed_Q_maxes_MW[Q_max_id]
            
            #the id of the agent's bid
            Q_bid_id = act_n[lv_agent][Q_idx]
            #get the ratio corresponding to that id
            Q_ratio = self.allowed_Q_ratios[ Q_bid_id ]
            #calculate Q
            Q = Q_max * Q_ratio
            
            #agent's actual marginal cost
            MC_id = obs_n[lv_agent][MC_idx]
            MC = self.allowed_MCs_USDperMW[MC_id]
            #the P_ratios id
            P_bid_id = act_n[lv_agent][P_idx]
            #get corresponding P_ratio value
            P_ratio = self.allowed_P_ratios[ P_bid_id ]
            P = MC * P_ratio
            
            #package
            bid = (Q,P,lv_agent)
            
            #store
            bids_unsorted.append(bid)
        #add the agent's number to the bid so we can track it
        
        bids_sorted = sorted(bids_unsorted, key=lambda QPN: QPN[P_idx])
        bids_sorted = list(bids_sorted)
        
        #loop to clear the market
        Q_dispatched = 0
        
        m_idx=2
        b_idx=3
        meaningless_agent_idx=0
        m_id = obs_n[meaningless_agent_idx][m_idx]
        b_id = obs_n[meaningless_agent_idx][b_idx]
        m = self.allowed_demand_slopes_USDperMW[m_id]
        b = self.allowed_demand_intercepts_USD[b_id]
        
        P_demand = lambda Q: b - m * Q
        bid_Q_idx = 0
        bid_P_idx = 1

        #initialize
        P_clear = P_demand(0)

        #stores the bid and how much was accepted
        QPNQ_arr = []
        for next_bid in bids_sorted:
            Q_next = next_bid[bid_Q_idx]
            P_next = next_bid[bid_P_idx]
            
            assert P_next >= P_clear or abs(Q_dispatched) < 1e-3
            
            #case 1: we accept the whole block
            if P_demand(Q_dispatched + Q_next) > P_next: #if there is demand even after taking the whole bid,
                Q_dispatched += Q_next
                Q_accepted = Q_next
                P_clear = P_next
            
            #case 2: we can accept some, but not all of this bid, i.e. it's where D intersects S
            elif P_demand(Q_dispatched + 0) > P_next:
                # assert False
                #P(Q_dis) -m*dQ = P_next
                #(P(Q_dis) - P_next ) / m
                dQ = (P_demand(Q_dispatched) - P_next) / m
                Q_accepted = dQ
                assert dQ > 0 and dQ - Q_next <= 1e-4
                Q_dispatched += Q_accepted
                P_clear = P_next
                assert abs(P_demand(Q_dispatched) - P_clear) <=1e-4
            
            #case 3: we accept none of this bid
            else:
                assert P_demand(Q_dispatched) <= P_next
                assert P_next >= P_clear
                Q_accepted = 0
            
            QPNQ = (*next_bid, Q_accepted)
            QPNQ_arr.append(QPNQ)
        
        reward_n = self.calculate_profit(QPNQ_arr, P_clear, obs_n)
        
        return QPNQ_arr, P_clear, reward_n
    
    def bids_arr_to_xy(self, bids_arr):
        #bid format is (Q,P,N,Q)
        x = []
        y = []
        Q_accepted_idx = 3
        Q_bid_idx = 0
        P_idx = 1
        agent_idx = 2
        x.append(0)
        y.append(0)
        
        Q_dispatched = 0
        for QPNQ in bids_arr:
            x.append(Q_dispatched)
            P = QPNQ[P_idx]
            y.append(P)
            
            Q_dispatched += QPNQ[Q_bid_idx]
            
            x.append(Q_dispatched)
            y.append(P)
        
        return x,y
    
    def plot_bids(self):
        
        my_legend = []
        
        #use the utility function to convert these to xy values
        current_bids_arr = self.current_bids_arr
        x_actual, y_actual = self.bids_arr_to_xy(current_bids_arr)
        plt.plot(x_actual, y_actual,linestyle='-',color="blue")
        my_legend.append("Actual Bids")
        #now plot the demand curves
        x_max = self.demand_intercept_USD / self.demand_slope_unsigned_USDperMW
        #P0 -mx =0 --> x = P0/m
        num_pts = 100
        x_demand = np.linspace(start=0,stop=x_max,num=num_pts)
        y_demand = self.demand_intercept_USD - self.demand_slope_unsigned_USDperMW * x_demand
        plt.plot(x_demand,y_demand,linestyle="--", color="blue")
        my_legend.append("Demand")
        plt.axhline(self.current_P_clear, linestyle='dotted',color="blue")
        my_legend.append("Actual P_clear")
        
        if self.PLOT_MONOPOLY_OUTCOME:
            monopoly_bids_arr = self.monopoly_bids_arr
            x_monopoly, y_monopoly = self.bids_arr_to_xy(monopoly_bids_arr)
            plt.plot(x_monopoly, y_monopoly,linestyle='-',color="orange")
            my_legend.append("Monopoly Bids (= Supply Curve)")
            y_monopoly_demand = self.demand_intercept_USD - 2* self.demand_slope_unsigned_USDperMW * x_demand
            plt.plot(x_demand, y_monopoly_demand,linestyle="--", color="orange")
            my_legend.append("Monopoly Effective-Demand")
            plt.axhline(self.monopoly_P_clear,linestyle='dotted', color="orange")
            my_legend.append("Monopoly P_clear")
        if self.PLOT_AT_COST_OUTCOME:
            at_cost_bids_arr = self.at_cost_bids_arr
            x_at_cost, y_at_cost = self.bids_arr_to_xy(at_cost_bids_arr)
            plt.plot(x_at_cost, y_at_cost, linestyle='-',color="red")
            my_legend.append("At Cost Bids")
            plt.axhline(self.at_cost_P_clear,linestyle='dotted',color="red")
            my_legend.append("At Cost P_clear")
        
        plt.xlabel("Quantity MW")
        plt.ylabel("Price/MW USD")
        plt.legend(my_legend,loc="best")
        
        y_max = max(self.allowed_MCs_USDperMW)*max(self.allowed_P_ratios)*1.1
        y_max = max(y_max, self.demand_intercept_USD)
        y_max *= 1.1
        y_min = -0.1 * y_max
        my_ylims = [y_min,y_max]
        plt.ylim(my_ylims)
        
        x_max =  max(self.allowed_Q_maxes_MW)*self.num_agents * 1.1
        
        x_min = -0.1 * x_max
        my_xlims =[x_min, x_max]
        plt.xlim(my_xlims)
        
        plt.show()
        return
    
    def reset(self):
        raise NotImplementedError
        
    def render(self,):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
        
    def calculate_monopoly_outcome(self):
        #clears the market as if agents were plants controlled by a single agent
        #NB does NOT respect the tabular action-constraints, does respect MCs and Q_maxes
        
        # P_D'(Q)*Q + P_D = MC (Wikipedia)
        # dispatch until MC = b - 2mQ
        
        self.agent_MCs_USDperMW
        self.agent_Q_maxes_MW
        QPN_list = []
        #list of (Q,MC, agent_id) pairs
        for lv in range(self.num_agents):
            tmp = (self.agent_Q_maxes_MW[lv], self.agent_MCs_USDperMW[lv], lv)
            #NB this is NOT a bid, since the price is MC
            QPN_list.append(tmp)
        
        MC_idx=1
        agent_idx = 2
        Q_idx=0
        QPN_list_sorted = sorted(QPN_list,key = lambda QPN: QPN[MC_idx])
        QPN_list_sorted = list(QPN_list_sorted)
        #sort by ascending marginal cost
        
        #dispatch until MC => b-2mQ
        P_monopoly = lambda Q: self.demand_intercept_USD - 2 * self.demand_slope_unsigned_USDperMW * Q
        Q_dispatched = 0
        lv = 0
        next_MC = QPN_list_sorted[lv][MC_idx]
        done = False
        bids_arr = []
        #QPNQ = (Q_max, MC, agent_id, Q_accepted)
        while not done:
            QPN = QPN_list_sorted[lv]
            Q_max_next = QPN[Q_idx]
            MC_next = QPN[MC_idx]
            
            if P_monopoly(Q_dispatched + Q_max_next) < next_MC:
                #if dispatching the whole next generator wouldn't 'recoup' marginal costs,
                Q_accepted = ( P_monopoly(Q_dispatched) - next_MC ) / (2*self.demand_slope_unsigned_USDperMW)
                assert Q_accepted < Q_max_next
                done = True
                Q_dispatched += Q_accepted
                P_clear = P_monopoly(Q_dispatched/2.0) #this is equivalent to P_demand(Q)
            else:
                Q_accepted = Q_max_next
                Q_dispatched += Q_accepted    
            
            
            tmp = (*QPN, Q_accepted) 
            bids_arr.append(tmp)
            
            lv+=1
            if lv >= self.num_agents:
                #if we reach shortage conditions
                done = True
                P_clear = P_monopoly(Q_dispatched/2.0) #this is equivalent to P_demand(Q)
        
        #add the rest with Q_accepted=0
        while lv < self.num_agents:
            QPN = QPN_list_sorted[lv]
            Q_accepted = 0
            tmp = (*QPN, Q_accepted)
            bids_arr.append(tmp)
            
            lv+=1
        
        
        self.monopoly_bids_arr = bids_arr
        self.monopoly_P_clear = P_clear
        return
    
    def calculate_at_cost_outcome(self):
        #clears the market as if agents all bid (Q_max, MC)
        #NB does NOT respect the tabular action-constraints, does respect MCs and Q_maxes
        
        # dispatch until MC = b - mQ
        
        self.agent_MCs_USDperMW
        self.agent_Q_maxes_MW
        QPN_list = []
        #list of (Q,MC, agent_id) pairs
        for lv in range(self.num_agents):
            tmp = (self.agent_Q_maxes_MW[lv], self.agent_MCs_USDperMW[lv], lv)
            #NB this is NOT a bid, since the price is MC
            QPN_list.append(tmp)
        
        MC_idx=1
        agent_idx = 2
        Q_idx=0
        QPN_list_sorted = sorted(QPN_list,key = lambda QPN: QPN[MC_idx])
        QPN_list_sorted = list(QPN_list_sorted)
        #sort by ascending marginal cost
        
        #dispatch until MC => b-2mQ
        P_demand = lambda Q: self.demand_intercept_USD - self.demand_slope_unsigned_USDperMW * Q
        Q_dispatched = 0
        lv = 0
        next_MC = QPN_list_sorted[lv][MC_idx]
        done = False
        bids_arr = []
        #QPNQ = (Q_max, MC, agent_id, Q_accepted)
        while not done:
            QPN = QPN_list_sorted[lv]
            Q_max_next = QPN[Q_idx]
            MC_next = QPN[MC_idx]
            
            if P_demand(Q_dispatched + Q_max_next) < next_MC:
                #if dispatching the whole next generator wouldn't recoup marginal costs,
                Q_accepted = ( P_demand(Q_dispatched) - next_MC ) / (self.demand_slope_unsigned_USDperMW)
                assert Q_accepted < Q_max_next
                done = True
                Q_dispatched+=Q_accepted
                P_clear = P_demand(Q_dispatched)
            else:
                Q_accepted = Q_max_next
                
            Q_dispatched += Q_accepted
            
            tmp = (*QPN, Q_accepted) 
            bids_arr.append(tmp)
            
            lv+=1
            if lv >= self.num_agents:
                #if we reach shortage conditions
                done = True
                P_clear = P_demand(Q_dispatched)
        
        #add the rest with Q_accepted=0
        while lv < self.num_agents:
            QPN = QPN_list_sorted[lv]
            Q_accepted = 0
            tmp = (*QPN, Q_accepted)
            bids_arr.append(tmp)
            
            lv+=1
        
        
        self.at_cost_bids_arr = bids_arr
        self.at_cost_P_clear = P_clear
        return
    
    def calculate_monopoly_regret(self):
        #calculates the difference in reward as occurred vs what would occur if agents bid MC
        optimal_bids = self.monopoly_bids_arr
        optimal_P_clear = self.monopoly_P_clear
        obs_n = self.current_obs_n
        optimal_profit = self.calculate_profit(optimal_bids, optimal_P_clear, obs_n)
        
        actual_bids = self.current_bids_arr
        actual_P_clear = self.current_P_clear
        actual_profit = self.calculate_profit(actual_bids, actual_P_clear, obs_n)
        
        assert len(actual_profit) == len(optimal_profit)
        
        regret_n = [optimal_profit[lv] - actual_profit[lv] for lv in range(len(optimal_profit))]
        return regret_n
    
    def calculate_allowed_hidden_states(self):
        #returns a list of all possible states, modulo the publicly available demand information
        # (Q_maxes * MCs)^num_agents
        # used to calculate the conditionally optimal utility vectors (below)
        Q_maxes_list = list(range(self.num_allowed_Q_maxes))
        MC_list = list(range(self.num_allowed_MCs))
        
        #per-agent hidden state
        one_hidden_state = itertools.product(Q_maxes_list, MC_list)
        one_hidden_state = list(one_hidden_state)
        
        #overall hidden state
        allowed_hidden_states = itertools.product(one_hidden_state, repeat=self.num_agents)
        self.allowed_hidden_states = list(allowed_hidden_states)
        assert len(self.allowed_hidden_states) > 0
    
    def calculate_conditionally_optimal_utility_vector(self, agent_id, agent_obs):
        #calculates the expected-utility-vector for each agent, conditioned on holding the other agents' policies constant - assumes other agents are playing a DETERMINISTIC POLICY, otherwise I'd have to clear the market many many times
        # U such that pi_i * U_i = expected utility of agent i
        # how to account for different agents' observations?  ideally we should E-out all of the other agents' obs'
        
        assert self.DISCRETIZED
        assert len(list(self.allowed_hidden_states)) > 0
        
        single_action_space_size = self.num_allowed_Q_ratios * self.num_allowed_P_ratios
        
        utilities = np.zeros(shape=(single_action_space_size))

        #get the non-hidden part of the observation:
        b_idx = self.allowed_demand_intercepts_USD.index(self.demand_intercept_USD)
        m_idx = self.allowed_demand_slopes_USDperMW.index(self.demand_slope_unsigned_USDperMW)
        
        #since num_hidden_states should NOT account for the info _visible_ to the agent
        num_hidden_states = (self.num_allowed_Q_maxes * self.num_allowed_MCs) ** (self.num_agents - 1)
        
        #must average over all possible hidden states:
        for hidden_state_n in list(self.allowed_hidden_states):
            #hidden_state_n is n_agents cartesian product of Q_max and MC, [[Q1,MC1],[],...]
            #so one element is [[Q,MC], [Q,MC],...]
            
            obs_n = [(Q,MC,m_idx,b_idx) for (Q,MC) in hidden_state_n]
            
            #hacky way to skip over the ones that don't match agent's obs
            if not obs_n[agent_id] == agent_obs: continue

            act_n = [self.agents[lv].act_optimal(obs_n[lv]) for lv in range(self.num_agents)]

            #for each act this agent could take, calculate rwd
            for lv_act, act in enumerate(self.agents[agent_id].allowed_actions):
                act_n[agent_id] = act
                #clear market thus
                QPNQ_arr, P_clear, reward_n = self.clear_market(act_n, obs_n)
                #get this agent's profit
                rwd = reward_n[agent_id]
                #accumulate the reward since we will average it over possible hidden-state info
                utilities[lv_act] += rwd
                
        #normalise by number of observations -- ASSUMES UNIFORM OBSERVATIONS
        utilities /= num_hidden_states
                
        #assert abs(utilities[0]) > 1e-5
        #can be true if the full demand is met without this agent
        
        return utilities