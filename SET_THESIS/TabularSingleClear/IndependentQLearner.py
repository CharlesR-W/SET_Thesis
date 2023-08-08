#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:41:52 2023

@author: crw
"""

import numpy as np
import itertools
import random
from Learner import Learner
import gymnasium as gym
import copy

class IndependentTabularQLearner(Learner):
    
    def __init__(self, env, agent_id, explore_type, explore_param, rwd_max=1e5):
        self.env = env
        #the environment in which the agent is placed - for requesting info about state/action spaces mainly
        
        self.agent_id = agent_id
        #the agent's "ID" number, used for accessing arrays
        
        self.allowed_actions = self.get_allowed_actions()
        self.allowed_observations = self.get_allowed_observations()
        self.experience_buffer = { (o,a):[] for a in self.allowed_actions for o in self.allowed_observations}
        self.state_action_reward_table = { (o,a): rwd_max for a in self.allowed_actions for o in self.allowed_observations}

        
        self.action_space = self.env.action_space[self.agent_id]
        #this object is the gym.spaces object describing the agent's action space
        # looks like [(QP),...] with 'num_bids' of entries.
        
        assert self.env.DISCRETIZED
        #since this is a tabular learner, the environment must be discrete
        
        self.explore_type = explore_type
        #string naming the type of exploration to be used
        
        self.explore_param = explore_param
        #parameter modifying exploration behaviour, e.g. epsilon

        self.allowed_actions = self.get_allowed_actions()
        #NB this assumes all actionas are always allowed!
        
    def get_allowed_actions(self):
    #returns a list of all allowed actions the agent can take
        agent_space = self.env.action_space[self.agent_id]
        
        assert len(agent_space) == self.env.agent_bids_allowed
        # should have the n_bids of QP_pairs
        
        assert self.env.DISCRETIZED
        # just for good measure - still needs to be discretised
        
        assert np.all([isinstance(QP_pair[0],gym.spaces.Discrete) and isinstance(QP_pair[1],gym.spaces.Discrete) for QP_pair in agent_space.spaces])
        #action_space should be [(QP),(),()...]; 2 above is because it should be pair'
        
        
        allowed_Q_ratios = self.env.allowed_Q_ratios
        allowed_P_ratios = self.env.allowed_P_ratios
        all_single_bids = []
        for lv_Q, Q_ratio in enumerate(allowed_Q_ratios):
            for lv_P, P_ratio in enumerate(allowed_P_ratios):
                #create all a list of all allowed single bids (coded as integers)
                #looking like a = (0,0) a=(0,1) ...
                pair = (lv_Q, lv_P)
                all_single_bids.append(pair)
        
        all_bids_product = itertools.product(all_single_bids,repeat = self.env.agent_bids_allowed)
        #cartesian product of all possible single-bids
        
        ret = list(all_bids_product)
        #convert to list
        return ret
    
    def get_allowed_observations(self):
    #returns a list of all allowed actions the agent can take
        agent_space = self.env.observation_space[self.agent_id]
        
        assert self.env.DISCRETIZED
        # just for good measure - still needs to be discretised
        
        space_sizes = [s.n for s in agent_space.spaces]
        tmp_tuples = [tuple(range(x)) for x in space_sizes]
        all_obs = itertools.product(  *tmp_tuples )
        
        
        ret = tuple(all_obs)
        #convert to tuple
        return ret
        
    def act(self, obs):
    #determines the agent's next action based on the given observation (and expected rewards).  also implements exploration.
        
        if self.explore_type == "boltzmann":
        #Boltzmann exploration
            
            beta = self.explore_param
            #(inverse temperature)
            
            partition = 0
            #Partition function
            
            probabilities = []
            #list of prob for each action
            for lv, act in enumerate(self.allowed_actions):
            #calculate Boltzmann prob of each action, while tracking the partition func
                act_reward = self.state_action_reward_table[(obs,act)]
                #mean reward of (o,a)
                
                numerator = np.exp(act_reward*beta)
                #boltzmann numerator e^(beta*r)
                
                probabilities.append(numerator)
                
                partition += numerator
            probabilities = np.array(probabilities)
            probabilities /= partition
            
            act = np.random.choice(self.allowed_actions, p=probabilities)
            #select action based on calculated probabilities
            
        #epsilon exploration - choose a random action with probability epsilon    
        elif self.explore_type == "epsilon":
        #Epsilon exploration
            eps = self.explore_param
            rn = np.random.uniform(low=0.0,high=1.0)
            if rn <= eps:
                #choose randomly
                act = random.choice(self.allowed_actions)
            else:
                #choose optimally, breaking ties at random
                act = self.act_optimal(obs,self.allowed_actions)
        elif self.explore_type == "none":
        #No exploration
            act = self.act_optimal(obs,self.allowed_actions)
        
        elif self.explore_type == "uniform":
        #uniform random exploration
            act = random.choice(self.allowed_actions)
        
        return act
                    
    def act_optimal(self,obs, allowed_actions):
    #returns the idx of the action with the highest expected reward, breaking ties at random.
        
        reward_max = -float("inf")
        for lv, act in enumerate(allowed_actions):
        #foe each possible action,
            reward = self.state_action_reward_table[(obs,act)]
            #expected reward of (s,a)
            if reward > reward_max:
            #if better, replace optimal
                reward_max = reward
                lv_opt = [lv]
            if reward == reward_max:
            #if equal, store for tie-breaking
                lv_opt.append(lv)
            
        lv_opt = random.choice(lv_opt)
        #if there are multiple optimal actions, break the tie randomly; if there is only one optimal action, this is equivalent to returning that
        ret = self.allowed_actions[lv_opt]
        return ret
    
    def learn(self, obs, act, reward):
        self.store_experience(obs, act, reward)
        #add the experience to the experience buffer
        
        self.update_reward_tables(obs, act, reward)
        #update reward tables accordingly
    
    def store_experience(self,obs,act,reward):
    #stores reward for (obs, action) pairs
        assert (obs,act) in self.experience_buffer
        #assert act in self.experience_buffer[obs]
        #make sure the (o,a) pair is in the reward buffer already
        
        self.experience_buffer[(obs, act)].append(reward)
        #add this reward to the list of those obtained
        
    def update_reward_tables(self,obs,act,reward):
        self.state_action_reward_table[(obs,act)] = np.mean( self.experience_buffer[(obs,act)] )
        #update Q(o,a)
        
        #self.state_reward_table[obs] = [self.state_action_reward_table[obs][a] for a in self.allowed_actions]
        #update V(o)