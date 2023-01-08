from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

import time
import matplotlib.pyplot as plt
import gym
import gym_offload_autoscale
import random
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


# dqn
rewards_list = []
avg_rewards = []
rewards_time_list = []
avg_rewards_time_list = []
rewards_bak_list = []
avg_rewards_bak_list = []
rewards_bat_list = []
avg_rewards_bat_list = []
avg_rewards_energy_list = []
dqn_data = []

train_time_slots = 1000
t_range = 2000


'''
    Here we implemented a DQN algorithm to compare with PPO.
    * We use a single hidden layer neural network.
    * The implementation is a stub, tbh.
    
'''


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = 1.0

        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=1000)  # 1000000
        # the neural network
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
    # remember, for exploitation

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmin(q_values[0])

    def replay(self):
        if len(self.memory) < 20:
            return
        batch = random.sample(self.memory, 20)
        for state, action, reward, next_state, terminal in batch:
            q_upd = reward
            if not terminal:
                q_upd = (reward + 0.95 *
                         np.amin(self.model.predict(next_state)[0]))
            q_val = self.model.predict(state)
            q_val[0][action] = q_upd
            self.model.fit(state, q_val, verbose=0)
        self.exploration_rate *= 0.995  # exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate)


set_seed(rand_seed)
obs = env.reset()


def agent():
    env = gym.make('offload-autoscale-v0', p_coeff=x)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    solver = DQNSolver(observation_space, action_space)
    episode = 0
    accumulated_step = 0
    while True:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        terminal = False
        step = 0
        while True:
            done = False
            action = solver.act(state)
            next_state, reward, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            step += 1
            accumulated_step += 1
            print('\tstate: ', state)
            if step >= 96:  # Termination of a single episode
                done = True
            solver.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                episode += 1
                break
            solver.replay()
            # Termination of the entire training process.
            if accumulated_step == train_time_slots:
                terminal = True
                break
        if terminal:
            break
    print('episode: ', episode)
    for _ in range(t_range):
        action = solver.act(state)
        next_state, reward, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, observation_space])
        t, bak, bat = env.render()
        rewards_list.append(1 / reward)
        avg_rewards.append(np.mean(rewards_list[:]))
        rewards_time_list.append(t)
        avg_rewards_time_list.append(np.mean(rewards_time_list[:]))
        rewards_bak_list.append(bak)
        avg_rewards_bak_list.append(np.mean(rewards_bak_list[:]))
        rewards_bat_list.append(bat)
        avg_rewards_bat_list.append(np.mean(rewards_bat_list[:]))
        avg_rewards_energy_list.append(avg_rewards_bak_list[-1]+avg_rewards_bat_list[-1])
        dqn_data.append([avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]])


agent()

# print('--RESULTS--')
# print('{:15}{:30}{:10}{:10}{:10}'.format('method','total cost','time','bak','bat'))
# print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('dqn',avg_rewards[-1], avg_rewards_time_list[-1], avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]))
# end_time = time.time()
# print('elapsed time:', end_time-start_time)

# #total cost
# df=pd.DataFrame({'x': range(t_range), 'y_6': avg_rewards})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# plt.plot('x', 'y_6', data=df, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="Q Learning")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn/dqn_total_p='+str(x)+'_.xlsx'
# export_excel = df.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/dqn/dqn_total_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #time cost
# dft=pd.DataFrame({'x': range(t_range), 'y_6': avg_rewards_time_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Delay Cost")
# plt.plot('x', 'y_6', data=dft, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="Q Learning")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn/dqn_time_p='+str(x)+'_.xlsx'
# export_excel = dft.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/dqn/dqn_time_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #bak-up cost
# dfbak=pd.DataFrame({'x': range(t_range), 'y_6': avg_rewards_bak_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Back-up Power Cost")
# plt.plot('x', 'y_6', data=dfbak, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="Q Learning")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn/dqn_backup_p='+str(x)+'_.xlsx'
# export_excel = dfbak.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/dqn/dqn_backup_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #battery cost
# dfbat=pd.DataFrame({'x': range(t_range),'y_6': avg_rewards_bat_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Battery Cost")
# plt.plot('x', 'y_6', data=dfbat, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="Q Learning")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn/dqn_battery_p='+str(x)+'_.xlsx'
# export_excel = dfbat.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/dqn/dqn_battery_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
# #energy cost
# dfe=pd.DataFrame({'x': range(t_range), 'y_6': avg_rewards_energy_list})
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Energy Cost")
# plt.plot('x', 'y_6', data=dfe, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="Q Learning")
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn/dqn_energy_p='+str(x)+'_.xlsx'
# export_excel = dfe.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'p='+str(x)+'/dqn/dqn_energy_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# #dqn area chart
# plt.xlabel("Time Slot")
# plt.ylabel("Average Costs")
# xx = range(t_range)
# yy = [avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list]
# fig = plt.stackplot(xx, yy, colors = 'w', edgecolor = 'black', labels = ['Delay cost', 'Backup cost', 'Battery cost'])
# hatches = ['...', '+++++', '///']
# for s, h in zip(fig, hatches):
#     s.set_hatch(h)
# plt.title('DQN')
# plt.legend()
# plt.grid()
# my_file = 'p='+str(x)+'/dqn_area'+'p='+str(x)+'.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
