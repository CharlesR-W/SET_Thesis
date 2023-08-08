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

def do_all_that_once(num_agents=1, num_episodes=int(1e5),num_Q_maxes=1, num_MCs=1,explore_type="epsilon", explore_param=1.0, update_type="soft", update_parameter=0.99, num_Q_ratios=5, num_P_ratios=5,data_collect_timestep = 128):    
    #some calculations we only want to do once in a while since they're too expensive to do at all steps
    
    
    
    param_dict = {
        #General environment parameters
        "num_agents":num_agents,
        "agent_bids_allowed":1,
        "Q_MAX_MW":200,
        "P_MAX_USD":500,
        "DISCRETIZED":True,
        "train_rounds": num_episodes,
    #    "test_rounds":int(1e3),
        "PLOT_AT_COST_OUTCOME": False,
        "CALCULATE_REGRET":False,
        "PLOT_MONOPOLY_OUTCOME": False,
        
        #Q_maxes
        "num_allowed_Q_maxes":num_Q_maxes,
        "Q_max_max": 101,
        "Q_max_min": 51,
        
        #marginal costs
        "num_allowed_MCs":num_MCs,
        "MC_max":50,
        "MC_min":20,
        
        #demand slope
        "num_allowed_demand_slopes":1,
        "demand_slope_max":2, #USD/MW
        "demand_slope_min":2,
        
        #demand intercept
        "num_allowed_demand_intercepts":1,
        "demand_intercept_max":500,
        "demand_intercept_min":500,
        
        #Q_ratios
        "num_allowed_Q_ratios":num_Q_ratios,
        "Q_ratio_max":1.0,
        "Q_ratio_min":0.5,
        
        #P_ratios
        "num_allowed_P_ratios":num_P_ratios,
        "P_ratio_max":10,
        "P_ratio_min":0.5,
        }
    start_exploit_round = param_dict["train_rounds"] / 2.0
    start_exploit_round = int(start_exploit_round)
    
    da_env = SingleBidDayAheadMarketEnvironment(param_dict)
    
    learner_arr = [SingleBidIndependentTabularQLearner(env=da_env, agent_id=lv, explore_type=explore_type, explore_param=explore_param, update_type=update_type, update_parameter=update_parameter, pessimism_parameter=0.0) for lv in range(num_agents)]
    
    done_n = [False]
    obs_n = da_env.get_observation_n()
    time = -1
    time_max = param_dict["train_rounds"]
    
    #records agent TD error over time - to measure convergence
    TD_errors = np.zeros(shape=(param_dict["num_agents"], time_max))
    
    #measures agents' regret heuristic over time
    #regrets = np.zeros(shape=(param_dict["num_agents"], time_max))
    
    #measures how far short agent is of playing best-response; calculate rarely since its compute-intensive
    deficits = np.zeros(shape=(num_agents, time_max // data_collect_timestep))
    while not np.any(done_n):
        time += 1
        act_n = [learner.act(obs_n[lv]) for lv, learner in enumerate(learner_arr)]
        
        predicted_rewards = [learner.state_action_reward_table[(obs_n[lv],act_n[lv])] for lv, learner in enumerate(learner_arr)]
        #calculate each agent's predicted reward
        
        act_n, obs_n, reward_n, done_n, info_n = da_env.step(act_n)
        #act_n is the just-completed act, and obs_n is only redrawn at the end
        
        if time % data_collect_timestep == 0 and time >0:
            print(f"Timesetp: {time}")
            #assert False
            #da_env.plot_bids()
            #da_env.plot_bids()
            #assert False
            #every few steps, calculate policy deficits for the specified agent
            for agent_id in range(num_agents):
                agent_obs = obs_n[agent_id]
                
                utilities = da_env.calculate_conditionally_optimal_utility_vector(agent_id, agent_obs)
                
                idx_t = time // data_collect_timestep - 1
                
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
                deficits[agent_id,idx_t] = deficit
            
        if time >= start_exploit_round:
            for learner in learner_arr:
                learner.explore_param = 0 #no more exploring
                #learner.update_parameter = 1.0 #no more updating
        #breakpoint()
        for lv_agent in range(num_agents):
            obs = obs_n[lv_agent]
            rwd = reward_n[lv_agent]
            act = act_n[lv_agent]
            
            learner_arr[lv_agent].learn(obs,act,rwd)
            
            TD_error = rwd - predicted_rewards[lv_agent]
            TD_errors[lv_agent, time] = TD_error
    
            #regret = regret_n[lv_agent]
            #regrets[lv_agent, time] = regret
        
        obs_n = da_env.update_new_round()
        
    Q_tables = torch.zeros(size=[num_agents,num_Q_ratios,num_P_ratios])
    for lv_agent in range(num_agents):
        for lv_Q in range(num_Q_ratios):
            for lv_P in range(num_P_ratios):
                Q_tables[lv_agent, lv_Q, lv_P] = learner_arr[lv_agent].state_action_reward_table[ ((0,0,0,0),(lv_Q,lv_P) ) ]
    return TD_errors, deficits, Q_tables
#%%
#Plotting

"""num_actions = param_dict["num_allowed_Q_ratios"] * param_dict["num_allowed_P_ratios"]
num_states = param_dict["num_allowed_Q_maxes"] * param_dict["num_allowed_MCs"] * param_dict["num_allowed_demand_intercepts"] * param_dict["num_allowed_demand_slopes"]"""


num_runs=3
#num_agents=3
num_episodes=int(1e5)
#num_Q_maxes=2
#num_MCs=2
explore_type="epsilon"
explore_param=1.0
update_type="soft"
update_parameter=0.99
num_Q_ratios=3
num_P_ratios=10
data_collect_timestep=128

num_cases = 4
#num_quantities = 2 #mu,sigma b/t runs
import torch
case_deficits = torch.zeros(size=[num_cases,num_runs])#,num_quantities])
case_TDs = torch.zeros(size=[num_cases,num_runs])#,num_quantities])

lv_case = -1
for num_agents in [1,3]:
#    for num_Q_maxes, num_MCs in zip([2],[2]):
    for num_Q_maxes, num_MCs in zip([1,2],[1,2]):
        
        lv_case+=1
        
        num_batches = num_episodes // data_collect_timestep
        
        num_actions = num_Q_ratios*num_P_ratios
        num_states=num_Q_maxes*num_MCs
               
        TD_errors = torch.zeros(size=[num_runs,num_agents,num_episodes])
        deficits = torch.zeros(size=[num_runs,num_agents, num_batches])
        for lv_run in range(num_runs):
            tmp_TD_errors, tmp_deficits, tmp_Q_tables = do_all_that_once(
                num_agents=num_agents,
                num_episodes=num_episodes,
                num_Q_maxes=num_Q_maxes,
                num_MCs=num_MCs,
                explore_type=explore_type,
                explore_param=explore_param,
                update_type=update_type,
                update_parameter=update_parameter,
                num_Q_ratios=num_Q_ratios,
                num_P_ratios=num_P_ratios,
                data_collect_timestep=data_collect_timestep,
                )
            TD_errors[lv_run,:,:] = torch.from_numpy(tmp_TD_errors)
            deficits[lv_run,:,:] = torch.from_numpy(tmp_deficits)
        
        for lv_plot in range(2):
            window_size = 50
            if lv_plot == 0:
                suptitle="(Absolute) Agent TD errors over time (Smoothed)"
                x=range(num_episodes-window_size)
                y=torch.abs(TD_errors)
                
                xlabel="Episode"
                ylabel="TD Error"
            else:
                #breakpoint()
                suptitle="Deficit of agent(s) from conditional best-response (Smoothed)"
                x=range(num_batches-window_size)
                y=deficits
                xlabel = f"Batch (batch-size={data_collect_timestep})"
                ylabel="Deficit (USD)"
            
            #mu and sigma for bar chart
            final_window_size_episodes = 1000
            final_window_size_batches = final_window_size_episodes //data_collect_timestep
            if lv_plot==0:
                tmp = TD_errors[:,:,-final_window_size_episodes:]
            else:
                tmp = deficits[:,:,-final_window_size_batches:]
            torch.abs_(tmp)
            ep_idx=2
            tmp = torch.mean(tmp,dim=ep_idx)
            agent_idx=1
            #tmp = torch.mean(tmp,dim=1)
            if lv_plot ==0:
                case_TDs[lv_case,:] = torch.mean(tmp, dim=agent_idx)
                #case_TDs[lv_case,:,1] = torch.std(tmp, dim=agent_idx)
            else:
                case_deficits[lv_case,:] = torch.mean(tmp, dim=agent_idx)
                #case_deficits[lv_case,:,1] = torch.std(tmp, dim=agent_idx)
                
            #smooth for plotting
            tmp = torch.zeros(size=y.shape)
            tmp = tmp[:,:,window_size:]
            for lv in range(tmp.shape[2]):
                tmp[:,:,lv] = torch.mean(y[:,:,lv:lv+window_size],dim=2)
            y=tmp
            
            #get mu and sigma
            agent_idx=1
            mu = torch.mean(y,dim=agent_idx)
            sigma = torch.std(y,dim=agent_idx)
                
            #now create plots
            style_label="seaborn-v0_8-bright"
            with plt.style.context(style_label):
                fig = plt.figure()
                for lv_run, sty_dict in zip(range(num_runs),plt.rcParams["axes.prop_cycle"]()):
                    color = sty_dict["color"]
                    
                    plt.plot(x, mu[lv_run], label=f"Run {lv_run}" + (" - Agent Average" if num_agents>1 else ""),color=color)
                    if num_agents>1:
                        plt.fill_between(x, mu[lv_run]-sigma[lv_run], mu[lv_run]+sigma[lv_run], alpha=0.2,color=color)
                #plt.ylim([y_bottom,y_top])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.suptitle(suptitle)
                plt.legend()
                title=f"Number of agents: {num_agents}, Number of actions: {num_actions}, Number of states: {num_states}"
                plt.title(title)
                
                filename = title.replace(" ","_").replace(":","").replace(",","") + ("TD" if lv_plot==0 else "deficit")
                plt.savefig(filename + ".svg")
#%%%
#now for the final bar chart:

style_label="seaborn-v0_8-bright"
with plt.style.context(style_label):
    fig,ax1 = plt.subplots()
    ax2=ax1.twinx()
    #bar spacings
    bar_width = 2
    inter_bar_space = 6
    artists_list=[]
    
    #center coordinates
    x_TD = [(bar_width+inter_bar_space)*lv_tmp for lv_tmp in range(num_cases)]
    x_def = [x_TD[lv_tmp] + bar_width for lv_tmp in range(len(x_TD))]

    for lv_bar, sty_dict in zip(range(2),plt.rcParams["axes.prop_cycle"]()): #2 = TD + def
        color = sty_dict["color"]
        
        run_dim=1
        label = "TD Error" if lv_bar==0 else "Deficit"
        if lv_bar==0:
            x=x_TD
            y = torch.mean(case_TDs, dim=run_dim)
            y_err = torch.std(case_TDs, dim=run_dim)
            
            s = lambda n: "s" if n > 1 else ""
            
            tick_labels = [f"{num_agents} Agent" +s(num_agents) + f",\n {num_states} State" + s(num_states) for num_agents in [1,3] for num_states in [1,4]]
            
            x_tick = [0.5*(x_TD[lv_tmp]+ x_def[lv_tmp]) for lv_tmp in range(len(x_TD))]
            
            kwargs = {"tick_label":tick_labels}
            ax = ax1
            ylabel = "TD Error"
            
        else:
            x=x_def
            
            scale = torch.mean(case_TDs).item() / torch.mean(case_deficits).item()
            print(f"scale: {scale}")
            
            y = torch.mean(case_deficits, dim=run_dim) * scale
            y_err = torch.std(case_deficits, dim=run_dim) * scale
            #tick_labels =[""] * num_cases
            #tick_locations = [None] * num_cases
            ax = ax2
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            ylabel="Deficit (USD)"
        
        bar = ax.bar(x,y,color=color, label=label, width=bar_width, **kwargs)
        artists_list.append(bar)

        err_art = ax.errorbar(x, y, y_err,fmt="o",color="k")
        ax.set_ylabel(ylabel)
    ax1.legend(handles=artists_list + [err_art],labels=["TD Error", "Deficit", "Inter-run Standard Deviation"], loc="upper left")
    ax1.set_xticks(ticks=x_tick, labels=tick_labels,)
        
    title=f"TD Error and Deficit for Discrete-Environment Cases"
    plt.title(title)
    
    filename = title.replace(" ","_").replace(":","").replace(",","") + ("TD" if lv_plot==0 else "deficit")
    plt.savefig(filename + ".svg")