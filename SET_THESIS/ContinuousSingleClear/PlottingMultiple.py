from copy import deepcopy
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

#objects of interest are 2-4 loss histories, reward_trajs, 2-4 nets, deficit history if DDPG

def get_reward_history(learner):
    
    reward_history = []
    
    for tran in learner.memory.memory:
        r = tran.reward
        reward_history.append(r)
    
    return reward_history

def get_agent_data(num_agents, is_MADDPG, num_varying_params, run_id):
    assert num_agents in [1,2,4]
    assert is_MADDPG in [True,False]
    assert num_varying_params in [0,1,2]
    assert run_id in range(0,5)
    
    run_name = f"num_agents_{num_agents}_" + f"_MADDPG_{is_MADDPG}_" + f"_num_env_variables_{num_varying_params}_" + f"_run_id_{run_id}_"
    
    dir_name = os.path.join(os.getcwd(), run_name)
    
    file_name = os.path.join(dir_name, "learner_arr") + ".pkl"
    
    #breakpoint()
    
    with open(file_name,"rb") as read_file:
        print(f"Started Loading run data from run_name : {run_name}")
        learner_arr = pickle.load(read_file)
        #print(f"Finished Loading run data from run_name : {run_name}")
        
        if is_MADDPG:
            policy_loss_histories = []
            critic_loss_histories = []
            reward_histories = []
            for learner in learner_arr:
                policy_loss_histories.append(learner.MADDPG_policy_loss_history)
                critic_loss_histories.append(learner.MADDPG_critic_loss_history)
                
                reward_history = get_reward_history(learner)
                reward_histories.append(reward_history)
                
            ret = (
                policy_loss_histories,
                critic_loss_histories,
                reward_histories,
                )
                
        else:
            DDPG_policy_loss_histories = []
            DDPG_critic_loss_histories = []
            MADDPG_policy_loss_histories = []
            MADDPG_critic_loss_histories = []
            deficit_wrt_final_histories = []
            deficit_wrt_actual_histories = []
            reward_histories = []
            for learner in learner_arr:
                DDPG_policy_loss_histories.append(learner.DDPG_policy_loss_history)
                DDPG_critic_loss_histories.append(learner.DDPG_critic_loss_history)
                
                MADDPG_policy_loss_histories.append(learner.MADDPG_policy_loss_history)
                MADDPG_critic_loss_histories.append(learner.MADDPG_critic_loss_history)
                
                deficit_wrt_final_histories.append(learner.deficit_wrt_final_arr)
                deficit_wrt_actual_histories.append(learner.deficit_wrt_actual_arr)
                
                reward_history = get_reward_history(learner)
                reward_histories.append(reward_history)
            
            ret = (
                DDPG_policy_loss_histories,
                DDPG_critic_loss_histories,
                MADDPG_policy_loss_histories,
                MADDPG_critic_loss_histories,
                deficit_wrt_final_histories,
                deficit_wrt_actual_histories,
                reward_histories,
                )
            
            return ret

#%%
"""
for num_agents in [1,2,4]:   
    
    for is_MADDPG in [True,False]:
        
        for lv_num_varying_params in [0,1,2]:

            for lv_run_id in range(5):
                
                tmp = get_agent_data(num_agents,is_MADDPG,lv_num_varying_params,lv_run_id)
                assert False
"""
tmp = get_agent_data(num_agents=1,is_MADDPG=True,num_varying_params=0,run_id=1)
#%% Things to plot
#1. Compare DDPG/MADDPG loss histories _within_ learners
    # compare Q-loss
    # compare Q_MADDPG(pi_DDPG) and Q_MADDPG(pi_MADDG)
#2. Compare reward trajectories

###BEGIN PLOT TEMPLATE
# DATA TO PLOT

#Plot 1: Q-losses (Q_DDPG and Q_MADDPG) for 1, 4 agent cases
#("how much better is Q_MADDPG at predicting how good an action is?")

#Plot 2: deficit_wrt_actual AND wrt_final for N=1,4
#("How much better is the ('ideal') MADDPG action predicted to be vs. the DDPG one (the DDPG act at run time, or what it would do in the same situation as the final policy)")

#one_agent_DDPG_critic_losses = np.zeros(shape=[5,int(1e5)])
#four_agent_DDPG_critic_losses = np.zeros(shape=[5,int(1e5)])

for num_agents in [1,4]:   
    for is_MADDPG in [False]:
        for lv_num_varying_params in [2]:
            for lv_run_id in range(5):
                
                data = get_agent_data(num_agents,is_MADDPG,lv_num_varying_params,lv_run_id)
                
                (
                    DDPG_policy_loss_histories,
                    DDPG_critic_loss_histories,
                    MADDPG_policy_loss_histories,
                    MADDPG_critic_loss_histories,
                    deficit_wrt_final_histories,
                    deficit_wrt_actual_histories,
                    reward_histories,
                ) = data
                
                if num_agents == 1:
                    tmp = np.asarray(DDPG_critic_loss_histories)
                    one_agent_DDPG_critic_losses = np.mean(tmp,axis=0)
                    
                    tmp = np.asarray(MADDPG_critic_loss_histories)
                    one_agent_MADDPG_critic_losses = np.mean(tmp,axis=0)
                elif num_agents == 4:
                    tmp = np.asarray(DDPG_critic_loss_histories)
                    four_agent_DDPG_critic_losses = np.mean(tmp,axis=0)
                    
                    tmp = np.asarray(MADDPG_critic_loss_histories)
                    four_agent_MADDPG_critic_losses =  np.mean(tmp,axis=0)
                else:
                    assert False

y1 = one_agent_DDPG_critic_losses
y2 = one_agent_MADDPG_critic_losses
y3 = four_agent_DDPG_critic_losses
y4 = four_agent_MADDPG_critic_losses
x = list(range(len(y1)))
#x = np.linspace(xstart, xstop, npts=len(y))

# PLOTTING PARAMETERS
x_min = min(x)
x_max = max(x)*1.1
y_min = min(y1)
y_max = max(y1)*1.1
x_label = "Episode"
y_label = "Critic Loss"
plot_title = "MADDPG and DDPG critic losses, N_agents=1,4"
plot_subtitle = "(num_env_params=2, averaged over 5 runs, and over the four agents in 4's case')"

# PLOT COMMANDS
plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.suptitle(plot_title)
plt.title(plot_subtitle)
plt.legend(["DDPG, N=1","MADDPG, N=1","DDPG, N=4","MADDPG, N=4"])
### END PLOT TEMPLATE