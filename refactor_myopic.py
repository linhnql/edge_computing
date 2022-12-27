from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

import time
import matplotlib.pyplot as plt
import gym
import gym_offload_autoscale
import numpy as np
import pandas as pd
import os

# start_time = time.time()
my_path = os.path.abspath('res')


def set_seed(rand_seed):
    set_global_seeds(100)
    env.env_method('seed', rand_seed)
    np.random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    model.set_random_seed(rand_seed)


rand_seed = 1234
x = 0.5

env = gym.make('offload-autoscale-v0', p_coeff=x)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1, seed=rand_seed)
model.learn(total_timesteps=1000)


# myopic
rewards_list = []
avg_rewards = []
rewards_time_list = []
avg_rewards_time_list = []
rewards_bak_list = []
avg_rewards_bak_list = []
rewards_bat_list = []
avg_rewards_bat_list = []
avg_rewards_energy_list = []
myopic_data = []


train_time_slots = 20000
t_range = 2000

'''
    Myopic algorithm: The calculation to get the next action is implemented in the environment file.
    * Myopic is simply a greedy approach. We will use the optimization of scipy (namely minimize_scalar)
      to get the minimum value of the power function which corresponds to current state of the environment,
      and we take that action.
    * Myopic just optimize for a single timeslot.
'''

set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('myopic_action_cal')
    obs, rewards, dones, info = env.step(action)
    rewards_list.append(1 / rewards)
    avg_rewards.append(np.mean(rewards_list[:]))
    t, bak, bat = env.render()
    rewards_time_list.append(t)
    avg_rewards_time_list.append(np.mean(rewards_time_list[:]))
    rewards_bak_list.append(bak)
    avg_rewards_bak_list.append(np.mean(rewards_bak_list[:]))
    rewards_bat_list.append(bat)
    avg_rewards_bat_list.append(np.mean(rewards_bat_list[:]))
    avg_rewards_energy_list.append(avg_rewards_bak_list[-1]+avg_rewards_bat_list[-1])
    myopic_data.append([avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]])
    if dones:
        env.reset()


# print('--RESULTS--')
# print('{:15}{:30}{:10}{:10}{:10}'.format('method','total cost','time','bak','bat'))
# print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('myopic',avg_rewards[-1], avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]))
# end_time = time.time()
# print('elapsed time:', end_time-start_time)

# #total cost
# df=pd.DataFrame({'x': range(t_range), 'y_3': avg_rewards})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# plt.plot('x', 'y_3', data=df, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic/myopic_total_p='+str(x)+'_.xlsx'
# export_excel = df.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/myopic/myopic_total_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #time cost
# dft=pd.DataFrame({'x': range(t_range), 'y_3': avg_rewards_time_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Delay Cost")
# plt.plot('x', 'y_3', data=dft, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic/myopic_time_p='+str(x)+'_.xlsx'
# export_excel = dft.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/myopic/myopic_time_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #bak-up cost
# dfbak=pd.DataFrame({'x': range(t_range), 'y_3': avg_rewards_bak_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Back-up Power Cost")
# plt.plot('x', 'y_3', data=dfbak, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic/myopic_backup_p='+str(x)+'_.xlsx'
# export_excel = dfbak.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/myopic/myopic_backup_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #battery cost
# dfbat=pd.DataFrame({'x': range(t_range), 'y_3': avg_rewards_bat_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Battery Cost")
# plt.plot('x', 'y_3', data=dfbat, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic/myopic_battery_p='+str(x)+'_.xlsx'
# export_excel = dfbat.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/myopic/myopic_battery_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #energy cost
# dfe=pd.DataFrame({'x': range(t_range), 'y_3': avg_rewards_energy_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Energy Cost")
# plt.plot('x', 'y_3', data=dfe, marker='s', markevery = int(t_range/10), color='cyan', linewidth=1, label="myopic")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic/myopic_energy_p='+str(x)+'_.xlsx'
# export_excel = dfe.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/myopic/myopic_energy_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# #myopic area chart
# plt.xlabel("Time Slot")
# plt.ylabel("Average Costs")
# xx = range(t_range)
# yy = [avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list]
# fig = plt.stackplot(xx, yy, colors = 'w', edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
# hatches = ['...', '++++', '///']
# for s, h in zip(fig, hatches):
#     s.set_hatch(h)
# plt.title('Myopic')
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/myopic_area'+'p='+str(x)+'.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
