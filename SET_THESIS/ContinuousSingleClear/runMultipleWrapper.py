import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# Get CPU or GPU device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def learner_arr_to_plot_vars(learner_arr):
    policy_loss_histories = []
    critic_loss_histories = []
    supervisor_policy_loss_histories = []
    supervisor_critic_loss_histories = []
    supervisor_deficit_histories = []
    reward_histories = []
    bertrand_deficit_histories = []
    price_bid_histories = []
    
    #breakpoint()
    for learner in learner_arr:
        tmp = learner.policy_loss_history
        tmp = torch.tensor(tmp)
        policy_loss_histories.append(tmp)
        
        tmp = learner.critic_loss_history
        tmp = torch.tensor(tmp)
        critic_loss_histories.append(tmp)
        
        tmp = learner.supervisor_policy_loss_history
        tmp = torch.tensor(tmp)
        supervisor_policy_loss_histories.append(tmp)
        
        tmp = learner.supervisor_critic_loss_history
        tmp = torch.tensor(tmp)
        supervisor_critic_loss_histories.append(tmp)
        
        supervisor_deficit_histories.append(learner.deficit_history)
        
        reward_histories.append(learner.reward_history)
        
        if learner.BERTRAND:
            bertrand_deficit_histories.append(learner.bertrand_deficit_history)
        
            lv_agent = learner.agent_id
            price_idx = 1
            price_bid_history = []
            for tran in learner.memory.full_memory:
                price = tran.act[lv_agent, price_idx]
                price_bid_history.append(price)
                
            price_bid_history = torch.tensor(price_bid_history)
            price_bid_histories.append(price_bid_history)

    policy_loss_histories = torch.stack(policy_loss_histories)
    critic_loss_histories = torch.stack(critic_loss_histories)
    supervisor_policy_loss_histories = torch.stack(supervisor_policy_loss_histories)
    supervisor_critic_loss_histories = torch.stack(supervisor_critic_loss_histories)
    reward_histories = torch.stack(reward_histories)
    supervisor_deficit_histories = torch.stack(supervisor_deficit_histories)
    
    if learner.BERTRAND:
        bertrand_deficit_histories = torch.stack(bertrand_deficit_histories)
        price_bid_histories = torch.stack(price_bid_histories)
    
    if learner.BERTRAND:
        ret = (
            policy_loss_histories,
            critic_loss_histories,
            supervisor_policy_loss_histories,
            supervisor_critic_loss_histories,
            supervisor_deficit_histories,
            reward_histories,
            bertrand_deficit_histories,
            price_bid_histories,
            )
    else:
        ret = (
            policy_loss_histories,
            critic_loss_histories,
            supervisor_policy_loss_histories,
            supervisor_critic_loss_histories,
            supervisor_deficit_histories,
            reward_histories,
            )
    
    #breakpoint()
    
    return ret


random_seed=2776 #MMDCCLXXVI AUC

num_episodes = int(1e5)

#how many transitions should each gradient step account for?
batch_size = 128

#how many episodes between gradient updates
learn_interval = 128

default_state_distribution_dict = {
    "Q_max_distribution": [200,200],
    "MC_distribution": [40,40],
    "demand_slope_unsigned_distribution": [2,2],
    "demand_intercept_distribution": [500,500],
    }



default_agent_param_dict = {
    "learning_rate": 1e-3, 
    "discount_factor": 999,
    "memory_capacity": int(num_episodes/4),
    "explore_params": {
        "mu_state_noise": 0.,
        "mu_act_noise": 0.,
        "std_state_noise": 0.,
        "std_act_noise": 0.3,
        }
    }

from runContinuousDAMarket import perform_run
from copy import deepcopy
import os
#import pickle

tracked_history_names = [
    "policy_loss_histories",
    "critic_loss_histories",
    "supervisor_policy_loss_histories",
    "supervisor_critic_loss_histories",
    "supervisor_deficit_histories",
    "reward_histories",
    "bertrand_deficit_histories",
    "price_bid_histories",
    ]
plot_title_dict = [
    "Actor Policy Loss",
    "Actor Critic Loss",
    "Supervisor Policy Loss",
    "Supervisor Critic Loss",
    "Supervisor Estimate of Deficit",
    "Profit",
    "Deficit of Actor from Bertrand-Optimal",
    "Price Bid",
    ]
plot_title_dict = {key : plot_title_dict[lv] for lv,key in enumerate(tracked_history_names) }

plot_xlabel_dict = [f"Batch (batch-size={batch_size})"] * 4 + ["Episode"] * 4
plot_xlabel_dict = {key : plot_xlabel_dict[lv] for lv,key in enumerate(tracked_history_names) }

plot_ylabel_dict = ["Loss (arbitrary units)"] * 4 + [
    "Estimated Deficit (USD)",
    "Profit (USD)",
    "Estimated Deficit (USD)",
    "Bid (Normalized to Marginal Cost)",
    ]
plot_ylabel_dict = {key : plot_ylabel_dict[lv] for lv,key in enumerate(tracked_history_names) }


def runner_of_runners(num_learners, num_naive, is_MADDPG, num_varying_params, num_runs, BERTRAND=False, Q_max=300):
    
    if BERTRAND:
        tracked_histories_dict = {t:[] for t in tracked_history_names}
    else:
        tracked_histories_dict = {t:[] for t in tracked_history_names[:-2]}
    state_distribution_dict = deepcopy(default_state_distribution_dict)
    
    if num_varying_params == 0:
        state_distribution_dict["Q_max_distribution"] = [Q_max, Q_max]
        state_distribution_dict["MC_distribution"] = [40,40]
    else: 
        assert False
    """elif num_varying_params == 1:
        state_distribution_dict["Q_max_distribution"] = [100,200]
        state_distribution_dict["MC_distribution"] = [40,40]
    elif num_varying_params == 2:
        state_distribution_dict["Q_max_distribution"] = [100,200]
        state_distribution_dict["MC_distribution"] = [10,40]"""
    
    env_param_dict = {
        "state_distribution_dict" : state_distribution_dict,
        "num_learners": num_learners,
        "num_naive": num_naive,
        "num_agents": num_learners + num_naive,
        "reward_scale" : 1/1000,
        }
    agent_param_dict = deepcopy(default_agent_param_dict)
        
    for lv_run_id in range(num_runs):
        run_name = f"num_learners_{num_learners}_" +"num_naive_{num_naive}_"+ f"_MADDPG_{is_MADDPG}_" + f"_num_env_variables_{num_varying_params}_" + f"_run_id_{lv_run_id}_"
        
        #print(f"Beginning run for run_name : {run_name}")
        learner_arr = perform_run(env_param_dict=env_param_dict, agent_param_dict=agent_param_dict, learn_interval=learn_interval, batch_size=batch_size, num_episodes=num_episodes,  num_learners=num_learners, random_seed=random_seed, run_name=run_name, num_naive=num_naive, is_MADDPG=is_MADDPG, PLOT_BIDS=False, has_supervisor=True, BERTRAND=BERTRAND)

        for learner in learner_arr:
            learner.calc_reward_history()
            learner.calc_deficit_histories()

            
        data = learner_arr_to_plot_vars(learner_arr)
        
        #assert len(tmp) == len(tracked_history_names)
        for lv in range(len(data)):
            key = tracked_history_names[lv]
            tracked_histories_dict[key].append(data[lv])

    for lv in range(len(data)):
        key = tracked_history_names[lv]

        tmp = tracked_histories_dict[key]
        tmp = torch.stack(tmp)
        #print(f"storing for key: {key}, shape: {tmp.shape}")

        tracked_histories_dict[key] = tmp

    return tracked_histories_dict

def runner_of_plots(tracked_histories_dict, subtitle,save=True):
    
    lv_agent = 0
    lv_run = 0
    idx_agent=1

    for key in tracked_histories_dict.keys():
        #breakpoint()
        num_runs = int(tracked_histories_dict[key].shape[0])
        num_times = int(tracked_histories_dict[key].shape[2])
        y = tracked_histories_dict[key]
        window_size=50 if plot_xlabel_dict[key]=="Episode" else 1
        SMOOTHED = window_size>1
        y = rightward_window_smooth(y, window_size)
        num_agents = y.shape[idx_agent]
        mu = torch.mean(y, dim=idx_agent)
        sigma = torch.std(y, dim=idx_agent)
        
        if window_size == 1:
            x = list(range(num_times))
        else:
            x = list(range(num_times - window_size))
        # PLOTTING PARAMETERS
        x_min = min(x) - 0.1 * (max(x) - min(x))
        x_max = max(x) + 0.1 * (max(x) - min(x))
        y_min = torch.min(y) - 0.1 * (torch.max(y) - torch.min(y))
        y_max = torch.max(y) + 0.1 * (torch.max(y) - torch.min(y))
        x_label = plot_xlabel_dict[key]
        y_label = plot_ylabel_dict[key]
        plot_title = plot_title_dict[key] + (" (Smoothed)" if SMOOTHED else "")
        plot_subtitle = subtitle
        
        # PLOT COMMANDS
        style_label="seaborn-v0_8-bright"
        with plt.style.context(style_label):
            fig = plt.figure()
            for lv_run in range(num_runs):
                for lv_agent in range(num_agents):
                    pass#plt.plot(x,y.select(idx_agent,lv_agent)[lv_run],label="__None__",alpha=0.4)
                plt.plot(x,mu[lv_run], label=f"Run {lv_run}" + (" - Agent Average" if num_agents>1 else "") )
                plt.fill_between(x, mu[lv_run] - sigma[lv_run], mu[lv_run] + sigma[lv_run], alpha=0.2)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim([x_min,x_max])
            plt.ylim([y_min,y_max])
            plt.suptitle(plot_title)
            plt.title(plot_subtitle)
            plt.legend()
            if save:
                filename = plot_subtitle.replace(" ", "_").replace(",", "") + plot_title.replace(" ", "_").replace(",", "")
                plt.savefig(filename + ".svg")
                #plt.savefig(filename + ".fig")

def average_over_agents(histories):
    #assume shape = [lv_run, lv_agent, lv_episode]
    #x = histories #[num_runs, num_episodes]
    idx_agent=1
    mu = torch.mean(histories,dim=idx_agent)
    sigma = torch.std(histories,dim=idx_agent)
    
    return mu, sigma

def rightward_window_smooth(y, window_size=50):
    if window_size == 1:
        return y
    # history is [lv_episode]
    #window is NOT centered - it is to the RIGHT
    ret = torch.zeros(size=[y.shape[0],y.shape[1],y.shape[2]-window_size])
    idx_episode=2
    for lv in range(y.shape[idx_episode] - window_size):
        ret[:,:,lv] = torch.mean(y[:,:,lv:lv+window_size], dim=idx_episode)
    
    return ret

def get_final_window_average(histories, final_window_size):
    print(histories.shape)
    #breakpoint()
    idx_time=2 #?
    tmp = torch.mean(histories[:,:,-final_window_size:],dim=idx_time)
    idx_run=0
    idx_agent=1
    #FIRST avg over window, THEN, acg over agents
    tmp = torch.mean(tmp, dim=idx_agent)
    
    mu = torch.mean(tmp, dim=idx_run)
    sig = torch.std(tmp, dim=idx_run)

    mu = mu.item()
    sig = sig.item()
    
    return mu, sig

num_cases = 4
num_quantities = 2 #mu, sigma
final_window_size_episodes = 1000
final_window_size_batches = final_window_size_episodes // batch_size
end_critic_losses = torch.zeros(size=[num_cases, num_quantities])
end_deficits = torch.zeros(size=[num_cases, num_quantities])
lv_case = -1

#%% Does DDPG learn the monopoly when alone?  CASE 1a -- should be monopolist
# Meant to show the DDPG works at all in the continuous environment; Bertrand only so we can plot deficit
tracked_histories_dict = runner_of_runners(num_learners=1, num_naive=0, is_MADDPG=False, num_varying_params=0, num_runs=3, BERTRAND=True, Q_max=300)
subtitle = "1 DDPG Learner, 0 Naive Agents - Price-Only Bidding (Bertrand, Hypercompetitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["bertrand_deficit_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_deficits[lv_case,0] = mu
end_deficits[lv_case,1] = sig



runner_of_plots(tracked_histories_dict, subtitle,save=True)
print(f"Finished run with title: \n" + subtitle + "\n")
#%% Does a DDPG converge against a naive agent in Bertrand?  CASE 1b - should be competitive
tracked_histories_dict = runner_of_runners(num_learners=1, num_naive=1, is_MADDPG=False, num_varying_params=0, num_runs=3, BERTRAND=True, Q_max=300)
subtitle = "1 DDPG Learner, 1 Naive Agent - Price-Only Bidding (Bertrand, Hypercompetitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["bertrand_deficit_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_deficits[lv_case,0] = mu
end_deficits[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle,save=True)
print(f"Finished run with title: \n" + subtitle + "\n")
#%% Does a DDPG converge against DDPG in Bertrand?  CASE 1c - should be competitive
tracked_histories_dict = runner_of_runners(num_learners=2, num_naive=0, is_MADDPG=False, num_varying_params=0, num_runs=3, BERTRAND=True, Q_max=300)
subtitle = "2 DDPG Learners, 0 Naive Agents - Price-Only Bidding (Bertrand, Hypercompetitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["bertrand_deficit_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_deficits[lv_case,0] = mu
end_deficits[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle,save=True)
print(f"Finished run with title: \n" + subtitle + "\n")

#%% 1d MADDPG vs MADDPG
tracked_histories_dict = runner_of_runners(num_learners=2, num_naive=0, is_MADDPG=True, num_varying_params=0, num_runs=3, BERTRAND=True, Q_max=300)
subtitle = "2 MADDPG Learners, 0 Naive Agents - Price-Only Bidding (Bertrand, Hypercompetitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["bertrand_deficit_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_deficits[lv_case,0] = mu
end_deficits[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle,save=True)
print(f"Finished run with title: \n" + subtitle + "\n")
#%%
#now compare all 4 wrt critic-loss (~TD-error; NB, mean-squared) + Deficit also

#barchart time
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
        
        label = "TD Error" if lv_bar==0 else "Deficit"
        if lv_bar==0:
            x=x_TD
            y = end_critic_losses[:,0]
            y_err = end_critic_losses[:,1]
            
            s = lambda n: "s" if n > 1 else ""
            
            tick_labels = ["DDPG \nMonopolist", "DDPG vs \nNaive", "DDPG vs\n DDPG", "MADDPG vs\n MADDPG"]
            
            x_tick = [0.5*(x_TD[lv_tmp]+ x_def[lv_tmp]) for lv_tmp in range(len(x_TD))]
            
            kwargs = {"tick_label":tick_labels}
            ax = ax1
            ylabel = "Final Critic Loss (Unitless)"
            
        else:
            x=x_def
            scale=1
            #scale = torch.mean(end_critic_losses).item() / torch.mean(end_deficits).item()
            print(f"scale: {scale}")
            
            y = end_deficits[:,0] * scale
            y_err = end_deficits[:,1] * scale
            #tick_labels =[""] * num_cases
            #tick_locations = [None] * num_cases
            ax = ax2
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            bot, top = ax2.set_ylim(auto=True)
            
            #top=7000)
            #ax2.set_yticks(ticks=ax2.get_yticks(), labels=ax2.get_yticks()/scale)
            ylabel="Final Deficit (USD)"
        
        bar = ax.bar(x,y,color=color, label=label, width=bar_width, **kwargs)
        artists_list.append(bar)

        err_art = ax.errorbar(x, y, y_err,fmt="o",color="k")
        ax.set_ylabel(ylabel)
    ax1.legend(handles=artists_list + [err_art],labels=["Final, Agent-Averaged Critic Loss", "Final, Agent-Averaged Deficit", "Inter-run Standard Deviation"], loc="upper left")
    ax1.set_xticks(ticks=x_tick, labels=tick_labels,)
    
    ax2.set_ylim(bottom=0)
        
    title=f"Critic Loss and Deficit for Continuous, Bertrand Market"
    plt.title(title)
    
    filename = title.replace(" ","_").replace(",","")
    plt.savefig(filename + ".svg")



#%%

num_cases = 3
num_quantities = 2 #mu, sigma
final_window_size_episodes = 1000
final_window_size_batches = final_window_size_episodes // batch_size
end_critic_losses = torch.zeros(size=[num_cases, num_quantities])
end_profit = torch.zeros(size=[num_cases, num_quantities])
lv_case = -1

#%% Does a DDPG exploit a naive agent in full competition?  CASE 2a - should be monopolist-ish
tracked_histories_dict = runner_of_runners(num_learners=1, num_naive=1, is_MADDPG=False, num_varying_params=0, num_runs=3, BERTRAND=False, Q_max=200)
subtitle = "1 DDPG Learner, 1 Naive Agent - Q,P Bidding (Competitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["reward_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_profit[lv_case,0] = mu
end_profit[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle)
print(f"Finished run with title: \n" + subtitle + "\n")

#%% Does DDPG learn against each other in full competition? CASE 2b - should be fucky
tracked_histories_dict = runner_of_runners(num_learners=2, num_naive=0, is_MADDPG=False, num_varying_params=0, num_runs=3, BERTRAND=False, Q_max=200)
subtitle = "2 DDPG Learners, 0 Naive Agents - Q,P Bidding (Competitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["reward_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_profit[lv_case,0] = mu
end_profit[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle)
print(f"Finished run with title: \n" + subtitle + "\n")

#%% Does MADDPG fix the problem? CASE 2b - should be oligopoly
tracked_histories_dict = runner_of_runners(num_learners=2, num_naive=0, is_MADDPG=True, num_varying_params=0, num_runs=3, BERTRAND=False, Q_max=200)
subtitle = "2 MADDPG Learners, 0 Naive Agents - Q,P Bidding (Competitive)"

lv_case += 1
tmp = tracked_histories_dict["critic_loss_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_batches)
end_critic_losses[lv_case,0] = mu
end_critic_losses[lv_case,1] = sig

tmp = tracked_histories_dict["reward_histories"]
mu,sig = get_final_window_average(tmp, final_window_size_episodes)
end_profit[lv_case,0] = mu
end_profit[lv_case,1] = sig

runner_of_plots(tracked_histories_dict, subtitle,save=True)
print(f"Finished run with title: \n" + subtitle + "\n")

#%%
#now compare all 4 wrt critic-loss (~TD-error; NB, mean-squared) + Deficit also

#barchart time
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
        
        label = "TD Error" if lv_bar==0 else "Deficit"
        if lv_bar==0:
            x=x_TD
            y = end_critic_losses[:,0]
            y_err = end_critic_losses[:,1]
            
            s = lambda n: "s" if n > 1 else ""
            
            tick_labels = ["DDPG vs\nNaive", "DDPG vs \nDDPG", "MADDPG vs\n MADDPG"]
            
            x_tick = [0.5*(x_TD[lv_tmp]+ x_def[lv_tmp]) for lv_tmp in range(len(x_TD))]
            
            kwargs = {"tick_label":tick_labels}
            ax = ax1
            ylabel = "Final Critic Loss (Unitless)"
            
        else:
            x=x_def
            scale=1
            #scale = torch.mean(end_critic_losses).item() / torch.mean(end_deficits).item()
            print(f"scale: {scale}")
            
            y = end_profit[:,0] * scale
            y_err = end_profit[:,1] * scale
            #tick_labels =[""] * num_cases
            #tick_locations = [None] * num_cases
            ax = ax2
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            bot, top = ax2.set_ylim(auto=True)
            
            #top=7000)
            #ax2.set_yticks(ticks=ax2.get_yticks(), labels=ax2.get_yticks()/scale)
            ylabel="Final Profits (USD)"
            
            """
            b=500
            m=2
            c=40
            N=2
            cournot_q_per = (b-c) / (m*(N+1))
            cournot_p = 500 - 2*cournot_q_per * N
            cournot_profit_per = cournot_q_per * (cournot_p - c)
            
            ax2.scatter(x, [cournot_profit_per]*len(x), label="Cournot Oligopoly Profit",c="r",marker="x")
            """
            
        
        bar = ax.bar(x,y,color=color, label=label, width=bar_width, **kwargs)
        artists_list.append(bar)

        err_art = ax.errorbar(x, y, y_err,fmt="o",color="k")
        ax.set_ylabel(ylabel)

    
    
    ax1.legend(handles=artists_list + [err_art],labels=["Final, Agent-Averaged Critic Loss", "Final, Agent-Averaged Profit", "Inter-run Standard Deviation"], loc="upper left")
    ax1.set_xticks(ticks=x_tick, labels=tick_labels,)
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
        
    title=f"Critic Loss and Profit for Continuous, Q,P-Market"
    plt.title(title)
    
    filename = title.replace(" ","_").replace(",","")
    plt.savefig(filename + ".svg")