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


# fixed_0.4kW
rewards_list = []
avg_rewards = []
rewards_time_list = []
avg_rewards_time_list = []
rewards_bak_list = []
avg_rewards_bak_list = []
rewards_bat_list = []
avg_rewards_bat_list = []
avg_rewards_energy_list = []
fixed_1_data = []


train_time_slots = 20000
t_range = 2000


# Fixed the energy at 0.4kW
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
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
    fixed_1_data.append([avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]])
    if dones:
        env.reset()

# print('--RESULTS--')
# print('{:15}{:30}{:10}{:10}{:10}'.format('method','total cost','time','bak','bat'))
# print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('fixed 0.4kW',avg_rewards[-1], avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]))
# end_time = time.time()
# print('elapsed time:', end_time-start_time)

# #total cost
# df=pd.DataFrame({'x': range(t_range), 'y_4': avg_rewards})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# plt.plot('x', 'y_4', data=df, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed04/fixed04_total_p='+str(x)+'_.xlsx'
# export_excel = df.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/fixed04/fixed04_total_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #time cost
# dft=pd.DataFrame({'x': range(t_range),'y_4': avg_rewards_time_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Delay Cost")
# plt.plot('x', 'y_4', data=dft, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed04/fixed04_time_p='+str(x)+'_.xlsx'
# export_excel = dft.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/fixed04/fixed04_time_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #bak-up cost
# dfbak=pd.DataFrame({'x': range(t_range), 'y_4': avg_rewards_bak_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Back-up Power Cost")
# plt.plot('x', 'y_4', data=dfbak, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed04/fixed04_backup_p='+str(x)+'_.xlsx'
# export_excel = dfbak.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/fixed04/fixed04_backup_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #battery cost
# dfbat=pd.DataFrame({'x': range(t_range), 'y_4': avg_rewards_bat_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Battery Cost")
# plt.plot('x', 'y_4', data=dfbat, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed04/fixed04_battery_p='+str(x)+'_.xlsx'
# export_excel = dfbat.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/fixed04/fixed04_battery_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #energy cost
# dfe=pd.DataFrame({'x': range(t_range), 'y_4': avg_rewards_energy_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Energy Cost")
# plt.plot('x', 'y_4', data=dfe, marker='*', markevery = int(t_range/10), color='skyblue', linewidth=1, label="fixed 0.4kW")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed04/fixed04_energy_p='+str(x)+'_.xlsx'
# export_excel = dfe.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/fixed04/fixed04_energy_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# #fixed 0.4kW area chart
# plt.xlabel("Time Slot")
# plt.ylabel("Average Costs")
# xx = range(t_range)
# yy = [avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list]
# fig = plt.stackplot(xx, yy, colors = 'w', edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
# hatches = ['...', '+++++', '///']
# for s, h in zip(fig, hatches):
#     s.set_hatch(h)
# plt.title('Fixed 0.4kW')
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/fixed_0.4kW_area'+'p='+str(x)+'.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
