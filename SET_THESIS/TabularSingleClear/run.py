#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:42:42 2023

@author: crw
"""
from SingleBidDayAheadMarketEnvironment import SingleBidDayAheadMarketEnvironment
from SingleBidIndependentQLearner import SingleBidIndependentTabularQLearner
import numpy as np
import matplotlib.pyplot as plt

def avg_min_max(r, window_size):
    r_smoothed = [np.mean(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    r_min = [np.min(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    r_max = [np.max(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    
    return r_smoothed, r_min, r_max

num_agents = 2

#some calculations we only want to do once in a while since they're too expensive to do at all steps
data_collect_timestep = 1000


param_dict = {
    #General environment parameters
    "num_agents":num_agents,
    "agent_bids_allowed":1,
    "Q_MAX_MW":100,
    "P_MAX_USD":500,
    "DISCRETIZED":True,
    "train_rounds":int(1e5),
#    "test_rounds":int(1e3),
    "PLOT_AT_COST_OUTCOME": False,
    "CALCULATE_REGRET":True,
    "PLOT_MONOPOLY_OUTCOME": True,
    
    #Q_maxes
    "num_allowed_Q_maxes":1,
    "Q_max_max": 101,
    "Q_max_min": 51,
    
    #marginal costs
    "num_allowed_MCs":1,
    "MC_max":50,
    "MC_min":20,
    
    #demand slope
    "num_allowed_demand_slopes":1,
    "demand_slope_max":3, #USD/MW
    "demand_slope_min":1,
    
    #demand intercept
    "num_allowed_demand_intercepts":1,
    "demand_intercept_max":200,
    "demand_intercept_min":100,
    
    #Q_ratios
    "num_allowed_Q_ratios":5,
    "Q_ratio_max":1.0,
    "Q_ratio_min":0.5,
    
    #P_ratios
    "num_allowed_P_ratios":5,
    "P_ratio_max":2.0,
    "P_ratio_min":0.5,
    }

start_exploit_round = param_dict["train_rounds"] / 2.0
start_exploit_round = int(start_exploit_round)

da_env = SingleBidDayAheadMarketEnvironment(param_dict)

    learner_arr = [SingleBidIndependentTabularQLearner(env=da_env, agent_id=lv, explore_type="epsilon", explore_param=1.0, update_type="soft", update_parameter=0.99, pessimism_parameter=0.0) for lv in range(num_agents)]

done_n = [False]
observation_n = da_env.get_observation_n()
time = -1
time_max = param_dict["train_rounds"]

#records agent TD error over time - to measure convergence
TD_errors = np.zeros(shape=(param_dict["num_agents"], time_max))

#measures agents' regret heuristic over time
regrets = np.zeros(shape=(param_dict["num_agents"], time_max))

#measures how far short agent is of playing best-response; calculate rarely since its compute-intensive
deficits = np.zeros(shape=(time_max // data_collect_timestep))
while not np.any(done_n):
    time += 1
    act_n = [learner.act(observation_n[lv]) for lv, learner in enumerate(learner_arr)]
    
    predicted_rewards = [learner.state_action_reward_table[(observation_n[lv],act_n[lv])] for lv, learner in enumerate(learner_arr)]
    #calculate each agent's predicted reward
    
    act_n, observation_n, reward_n, regret_n, done_n, info_n = da_env.step(act_n)
    #act_n is the just-completed act, but observation_n is the new action
    #%%
    if time % data_collect_timestep == 0:
        print(f"Timesetp: {time}")
        #assert False
        #da_env.plot_bids()
        #da_env.plot_bids()
        #assert False
        #every few steps, calculate policy deficits for the specified agent
        agent_id = 0
        agent_obs = observation_n[agent_id]
        
        utilities = da_env.calculate_conditionally_optimal_utility_vector(agent_id, agent_obs)
        
        idx_t = time // data_collect_timestep
        
        agent_best_response_expected_profit = np.max(utilities)
        
        #what it THINKS is optimal
        agent_optimal_action = learner_arr[agent_id].act_optimal(agent_obs)
        agent_act = learner_arr[agent_id].allowed_actions.index(agent_optimal_action)
        
        #what it should expect to get if it acts this way
        agent_expected_profit = utilities[agent_act]
        deficit = agent_best_response_expected_profit - agent_expected_profit
        
        if time > start_exploit_round:
            #assert abs(deficit) < 1e-5
            pass
        
        #print(observation_n[0])
        
        assert deficit >= 0
        #
        deficits[idx_t] = deficit
        
    if time >= start_exploit_round:
        for learner in learner_arr:
            learner.explore_param = 0 #no more exploring
            learner.update_parameter = 1.0 #no more updating
    
    for lv_agent in range(num_agents):
        obs = observation_n[lv_agent]
        rwd = reward_n[lv_agent]
        act = act_n[lv_agent]
        
        learner_arr[lv_agent].learn(obs,act,rwd)
        
        TD_error = rwd - predicted_rewards[lv_agent]
        TD_errors[lv_agent, time] = TD_error

        regret = regret_n[lv_agent]
        regrets[lv_agent, time] = regret
    obs_n = da_env.update_new_round()
#%%
#Plotting

num_actions = param_dict["num_allowed_Q_ratios"] * param_dict["num_allowed_P_ratios"]
num_states = param_dict["num_allowed_Q_maxes"] * param_dict["num_allowed_MCs"] * param_dict["num_allowed_demand_intercepts"] * param_dict["num_allowed_demand_slopes"]


plt.figure()
window_size = 100
t = np.linspace(start=0,stop=time_max-1,num=time_max)
t = t[window_size*2:]

agent_idx=0
r = TD_errors[agent_idx,:]
r = np.abs(r)
r_smoothed, r_min, r_max = avg_min_max(r, window_size)
plt.plot( t,r_smoothed)
plt.fill_between(t,r_min,r_max,alpha=0.3)
#plt.ylim([y_bottom,y_top])
plt.xlabel("Time (rounds)")
plt.ylabel("Reward")
plt.suptitle("(Absolute) Agent TD errors over time")
plt.title(f"Number of agents: {num_agents}, Number of actions: {num_actions}, Number of states: {num_states}")
y_min = min(r_min)
y_max = max(r_max)
plt.vlines(start_exploit_round,ymin=y_min, ymax=y_max, color="red")

plt.figure()
r = np.mean(regrets,axis=0)

r_smoothed, r_min, r_max = avg_min_max(r,window_size)
r_smoothed, r_min, r_max = avg_min_max(r, window_size)
plt.plot( t,r_smoothed)
plt.fill_between(t,r_min,r_max,alpha=0.3)
#plt.ylim([y_bottom,y_top])
plt.xlabel("Time (rounds)")
plt.ylabel("Reward")
plt.suptitle("Agent [heuristic] regrets")
plt.title(f"Number of agents: {num_agents}, Number of actions: {num_actions}, Number of states: {num_states}")
y_min = min(r_min)
y_max = max(r_max)
plt.vlines(start_exploit_round,ymin=y_min, ymax=y_max, color="red")


#deficits = deficits[3:]

plt.figure()
plt.plot(np.linspace(start=0,stop=param_dict["train_rounds"], num=len(deficits)), deficits)
plt.suptitle("Deficit of agents from conditional best-response")
plt.title(f"Number of agents: {num_agents}, Number of actions: {num_actions}, Number of states: {num_states}")
plt.xlabel("Time")
plt.ylabel("Deficit ($)")
y_min = min(deficits)
y_max = max(deficits)
plt.vlines(start_exploit_round,ymin=y_min, ymax=y_max, color="red")