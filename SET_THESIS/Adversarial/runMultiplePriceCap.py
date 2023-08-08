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
    #supervisor_policy_loss_histories = []
    #supervisor_critic_loss_histories = []
    #supervisor_deficit_histories = []
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
        
        """tmp = learner.supervisor_policy_loss_history
        tmp = torch.tensor(tmp)
        supervisor_policy_loss_histories.append(tmp)
        
        tmp = learner.supervisor_critic_loss_history
        tmp = torch.tensor(tmp)
        supervisor_critic_loss_histories.append(tmp)"""
        
        #supervisor_deficit_histories.append(learner.deficit_history)
        
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
    #supervisor_policy_loss_histories = torch.stack(supervisor_policy_loss_histories)
    #supervisor_critic_loss_histories = torch.stack(supervisor_critic_loss_histories)
    reward_histories = torch.stack(reward_histories)
    #supervisor_deficit_histories = torch.stack(supervisor_deficit_histories)
    
    if learner.BERTRAND:
        bertrand_deficit_histories = torch.stack(bertrand_deficit_histories)
    price_bid_histories = torch.stack(price_bid_histories)
    
    if learner.BERTRAND:
        ret = (
            policy_loss_histories,
            critic_loss_histories,
            #supervisor_policy_loss_histories,
            #supervisor_critic_loss_histories,
            #supervisor_deficit_histories,
            reward_histories,
            bertrand_deficit_histories,
            price_bid_histories,
            )
    else:
        ret = (
            policy_loss_histories,
            critic_loss_histories,
            #supervisor_policy_loss_histories,
            #supervisor_critic_loss_histories,
            #supervisor_deficit_histories,
            reward_histories,
            price_bid_histories,
            )
    
    #breakpoint()
    
    return ret


random_seed=2776 #MMDCCLXXVI AUC

num_episodes = int(1e5)

#how many transitions should each gradient step account for?
batch_size = 128

#how many episodes between gradient updates
learn_interval = batch_size

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

from runPriceCapDA import perform_run
from copy import deepcopy
import os
#import pickle

tracked_history_names = [
    "policy_loss_histories",
    "critic_loss_histories",
#    "supervisor_policy_loss_histories",
#    "supervisor_critic_loss_histories",
#    "supervisor_deficit_histories",
    "reward_histories",
#    "bertrand_deficit_histories",
    "price_bid_histories",
#    "P_clear_histories"
    ]
plot_title_dict = [
    "Actor Policy Loss",
    "Actor Critic Loss",
#    "Supervisor Policy Loss",
#    "Supervisor Critic Loss",
#    "Supervisor Estimate of Deficit",
    "Profit / Social Welfare",
#    "Deficit of Actor from Bertrand-Optimal",
    "Clearing Price and Price Cap",
#    "NA"
    ]
plot_title_dict = {key : plot_title_dict[lv] for lv,key in enumerate(tracked_history_names) }

#plot_xlabel_dict = [f"Batch (batch-size={batch_size})"] * 4 + ["Episode"] * 4
plot_xlabel_dict = [f"Batch (batch-size={batch_size})"] * 2 + ["Episode"] * 2
plot_xlabel_dict = {key : plot_xlabel_dict[lv] for lv,key in enumerate(tracked_history_names) }

plot_ylabel_dict = ["Loss (arbitrary units)"] * 2 + [
    "Profit / Social Welfare (USD)",
    "Price (Normalized to Marginal Cost)",
    ]
plot_ylabel_dict = {key : plot_ylabel_dict[lv] for lv,key in enumerate(tracked_history_names) }


def runner_of_runners(num_learners, num_naive, is_MADDPG, num_varying_params, num_runs, alpha=0.5, BERTRAND=False, COURNOT=False, Q_max=300):
    
    tracked_histories_dict = {t:[] for t in tracked_history_names + ["P_clear_histories"]}
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
        "num_agents": num_learners + num_naive+1,
        "reward_scale" : 1/1000,
        "alpha" : alpha
        }
    agent_param_dict = deepcopy(default_agent_param_dict)
        
    for lv_run_id in range(num_runs):
        run_name = f"num_learners_{num_learners}_" +"num_naive_{num_naive}_"+ f"_MADDPG_{is_MADDPG}_" + f"_num_env_variables_{num_varying_params}_" + f"_run_id_{lv_run_id}_"
        
        #print(f"Beginning run for run_name : {run_name}")
        learner_arr, market_learner, P_clear_history = perform_run(env_param_dict=env_param_dict, agent_param_dict=agent_param_dict, learn_interval=learn_interval, batch_size=batch_size, num_episodes=num_episodes,  num_learners=num_learners, random_seed=random_seed, run_name=run_name, num_naive=num_naive, is_MADDPG=is_MADDPG, PLOT_BIDS=False, has_supervisor=True, BERTRAND=BERTRAND,COURNOT=COURNOT)

        for learner in learner_arr:
            learner.calc_reward_history()
            #learner.calc_deficit_histories()
        market_learner.calc_reward_history()
            
        data = learner_arr_to_plot_vars(learner_arr + [market_learner])
        #breakpoint()
        #assert len(tmp) == len(tracked_history_names)
        for lv in range(len(data)):
            key = tracked_history_names[lv]
            tracked_histories_dict[key].append(data[lv])
            
        tracked_histories_dict["P_clear_histories"].append(P_clear_history)

    for lv in range(len(data)):
        key = tracked_history_names[lv]

        tmp = tracked_histories_dict[key]
        tmp = torch.stack(tmp)
        #print(f"storing for key: {key}, shape: {tmp.shape}")

        tracked_histories_dict[key] = tmp
    
    tmp = tracked_histories_dict["P_clear_histories"]
    tmp = torch.stack(tmp)
    tracked_histories_dict["P_clear_histories"] = tmp
    
    return tracked_histories_dict

def runner_of_plots(tracked_histories_dict, subtitle,save=True):
    
    lv_agent = 0
    lv_run = 0
    idx_agent=1

    for key in tracked_histories_dict.keys():
        if key == "P_clear_histories":
            continue
        
        num_runs = int(tracked_histories_dict[key].shape[0])
        num_times = int(tracked_histories_dict[key].shape[2])
        y = tracked_histories_dict[key]
        window_size=50 if plot_xlabel_dict[key]=="Episode" else 1
        SMOOTHED = window_size>1
        y = rightward_window_smooth(y, window_size)
        
        y_market = y[:,-1].unsqueeze(idx_agent)
        y_firms = y[:,:-1]
        
        if key == "price_bid_histories":
            y_firms = tracked_histories_dict["P_clear_histories"] #since they only bid quantity anyways
            y_firms = y_firms.unsqueeze(idx_agent)
            y_firms = rightward_window_smooth(y_firms, window_size)
            
        num_learners = y_firms.shape[idx_agent]
        mu_firms = torch.mean(y_firms, dim=idx_agent)
        sigma_firms = torch.std(y_firms, dim=idx_agent)
        
        mu_market = torch.mean(y_market, dim=idx_agent)
        sigma_market = torch.std(y_market, dim=idx_agent)
        #sigma.to("cpu")
        
        if window_size == 1:
            x = list(range(num_times))
        else:
            x = list(range(num_times - window_size))
        # PLOTTING PARAMETERS
        x_min = min(x) - 0.1 * (max(x) - min(x))
        x_max = max(x) + 0.1 * (max(x) - min(x))

        y_min = torch.min(y)
        y_max = torch.max(y)
        dy = y_max - y_min
        y_min = y_min - 0.1 * dy
        y_max = y_max + 0.1 * dy

        x_label = plot_xlabel_dict[key]
        y_label = plot_ylabel_dict[key]
        plot_title = plot_title_dict[key] + (" (Smoothed)" if SMOOTHED else "")
        plot_subtitle = subtitle
        
        # PLOT COMMANDS
        style_label="seaborn-v0_8"
        num_agents = num_learners+1
        with plt.style.context(style_label):
            fig = plt.figure()
            for lv_run, sty_dict in zip(range(num_runs),plt.rcParams["axes.prop_cycle"]()):
                color = sty_dict["color"]
                
                for lv_firm in range(num_learners):
                    pass #plt.plot(x, y_firms[lv_run,lv_firm], label=f"Firm {lv_firm}", color=color, linestyle="solid")
                    
                for lv in range(2):
                    if lv == 0:
                        mu = mu_firms
                        sigma = sigma_firms
                        linestyle = "dotted"
                        label = f"Run {lv_run} - Agent" + " Average" if num_learners>1 else ""
                        if key == "price_bid_histories":
                            label = f"Run {lv_run} - Clearing Price"
                        else:
                            plt.fill_between(x, mu[lv_run] - sigma[lv_run], mu[lv_run] + sigma[lv_run], alpha=0.2, color=color)
                    else:
                        mu = mu_market
                        sigma = sigma_market
                        linestyle = "dashed"
                        label = f"Run {lv_run} - Market"
                        if key == "price_bid_histories":
                            label = f"Run {lv_run} - Price Cap"
                        
                    plt.plot(x, mu[lv_run], label=label, color=color, linestyle=linestyle)
                    
                    #plt.plot(x, mu[lv_run], label=label, color=color, linestyle=linestyle)
                    #plt.fill_between(x, mu[lv_run] - sigma[lv_run], mu[lv_run] + sigma[lv_run], alpha=0.2, color=color)
                
                
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim([x_min,x_max])
            plt.ylim([y_min,y_max])
            plt.suptitle(plot_title)
            plt.title(plot_subtitle)
            plt.legend()
            #breakpoint()
            if save:
                filename = plot_subtitle.replace(" ", "_").replace(",", "") + plot_title.replace(" ", "_").replace(",", "").replace("/","")
                plt.savefig(filename + ".png")
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

#%% 
N_arr = [1,3,5]
alpha_arr = [0.0,0.25,0.5,0.75,1.0]
num_runs = 3
final_cap = torch.zeros(size=[num_runs, len(N_arr), len(alpha_arr)])
final_clear = torch.zeros(size=[num_runs, len(N_arr), len(alpha_arr)])
for lv_N,N in enumerate(N_arr):
    for lv_alpha, alpha in enumerate(alpha_arr):
        tracked_histories_dict = runner_of_runners(num_learners=N, num_naive=0, is_MADDPG=True, num_varying_params=0, num_runs=num_runs, COURNOT=True, BERTRAND=False, Q_max=300,alpha=alpha)
        subtitle = f"N={N} alpha={alpha} Quantity-Only Bidding (Cournot) With Learned Price-Cap"
        runner_of_plots(tracked_histories_dict, subtitle, save=True)
        
        final_cap[:,lv_N, lv_alpha] = torch.mean(tracked_histories_dict["price_bid_histories"][:,-1,-1000:],dim=1)
        final_clear[:,lv_N, lv_alpha] = torch.mean(tracked_histories_dict["P_clear_histories"][:,-1000:],dim=1)
        
        print(f"Finished run with title: \n" + subtitle + "\n")

#%%
for lv in range(2):
    if lv == 0:
        p = final_cap
        #marker_sty = "x"
        suptitle = "Final Price Caps"
    else:
        p = final_clear
        #marker_sty = "."
        suptitle = "Final Clearing Prices"
    x = np.array(alpha_arr)
        
    style_label="seaborn-v0_8"
    with plt.style.context(style_label):
        fig = plt.figure()
        for lv_N, sty_dict in zip(range(len(N_arr)), plt.rcParams["axes.prop_cycle"]()):
            color = sty_dict["color"]
            
            N = N_arr[lv_N]
            y = (N+12.5)/(N+1) #oligopoly clearing price
            xmin,xmax = plt.xlim()
            plt.hlines(y,xmin=-0.3,xmax=1.3,color=color, label = f"N = {N} Oligopoly Price",linestyle="dotted")
            
            for lv_run in range(num_runs):
                y = p[lv_run,lv_N,:]
                plt.scatter(x,y,color=color,label = f"N = {N}" if lv_run ==0 else "__None__")
            y_mean = torch.mean(p[:,lv_N,:],dim=0)
            plt.plot(x,y_mean,color=color,label=f"N = {N} Run Average",linestyle="dashed")
            
        plt.xlabel("Alpha")
        plt.ylabel("Price (Normalized to Marginal Cost)")
        plt.suptitle(suptitle)
        plt.xlim(-0.1,1.1)
        plt.ylim(bottom=0.0)
        plt.title("Varying Adversary Preference and Number of Agents")
        filename = "AlphNP_Plot_" + "Cap" if lv ==0 else "Clear" + ".png"
        plt.legend()
        plt.savefig(filename + ".png")
                