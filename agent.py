from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
import openpyxl
import time
import matplotlib.pyplot as plt
import gym
import gym_offload_autoscale
import random
import numpy as np
import pandas as pd
import os
my_path = os.path.abspath('res')


start_time = time.time()


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

# ppo
rewards_list_ppo = []
avg_rewards_ppo = []
rewards_time_list_ppo = []
avg_rewards_time_list_ppo = []
rewards_bak_list_ppo = []
avg_rewards_bak_list_ppo = []
rewards_bat_list_ppo = []
avg_rewards_bat_list_ppo = []
avg_rewards_energy_list_ppo = []
ppo_data = []

# random
rewards_list_random = []
avg_rewards_random = []
rewards_time_list_random = []
avg_rewards_time_list_random = []
rewards_bak_list_random = []
avg_rewards_bak_list_random = []
rewards_bat_list_random = []
avg_rewards_bat_list_random = []
avg_rewards_energy_list_random = []
random_data = []

# myopic
rewards_list_myopic = []
avg_rewards_myopic = []
rewards_time_list_myopic = []
avg_rewards_time_list_myopic = []
rewards_bak_list_myopic = []
avg_rewards_bak_list_myopic = []
rewards_bat_list_myopic = []
avg_rewards_bat_list_myopic = []
avg_rewards_energy_list_myopic = []
myopic_data = []

# fixed_0.4kW
rewards_list_fixed_1 = []
avg_rewards_fixed_1 = []
rewards_time_list_fixed_1 = []
avg_rewards_time_list_fixed_1 = []
rewards_bak_list_fixed_1 = []
avg_rewards_bak_list_fixed_1 = []
rewards_bat_list_fixed_1 = []
avg_rewards_bat_list_fixed_1 = []
avg_rewards_energy_list_fixed_1 = []
fixed_1_data = []

# fixed_1kW
rewards_list_fixed_2 = []
avg_rewards_fixed_2 = []
rewards_time_list_fixed_2 = []
avg_rewards_time_list_fixed_2 = []
rewards_bak_list_fixed_2 = []
avg_rewards_bak_list_fixed_2 = []
rewards_bat_list_fixed_2 = []
avg_rewards_bat_list_fixed_2 = []
avg_rewards_energy_list_fixed_2 = []
fixed_2_data = []

# dqn
rewards_list_dqn = []
avg_rewards_dqn = []
rewards_time_list_dqn = []
avg_rewards_time_list_dqn = []
rewards_bak_list_dqn = []
avg_rewards_bak_list_dqn = []
rewards_bat_list_dqn = []
avg_rewards_bat_list_dqn = []
avg_rewards_energy_list_dqn = []
dqn_data = []

train_time_slots = 20000
t_range = 2000

'''
    Myopic algorithm: The calculation to get the next action is implemented in the environment file.
    * Myopic is simply a greedy approach. We will use the optimization of scipy (namely minimize_scalar)
      to get the minimum value of the power function which corresponds to current state of the environment,
      and we take that action.
    * Myopic just optimize for a single timeslot.
'''

# myopic
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('myopic_action_cal')
    obs, rewards, dones, info = env.step(action)
    rewards_list_myopic.append(1 / rewards)
    avg_rewards_myopic.append(np.mean(rewards_list_myopic[:]))
    t, bak, bat = env.render()
    rewards_time_list_myopic.append(t)
    avg_rewards_time_list_myopic.append(np.mean(rewards_time_list_myopic[:]))
    rewards_bak_list_myopic.append(bak)
    avg_rewards_bak_list_myopic.append(np.mean(rewards_bak_list_myopic[:]))
    rewards_bat_list_myopic.append(bat)
    avg_rewards_bat_list_myopic.append(np.mean(rewards_bat_list_myopic[:]))
    avg_rewards_energy_list_myopic.append(
        avg_rewards_bak_list_myopic[-1]+avg_rewards_bat_list_myopic[-1])
    myopic_data.append([avg_rewards_time_list_myopic[-1],
                       avg_rewards_bak_list_myopic[-1], avg_rewards_bat_list_myopic[-1]])
    if dones:
        env.reset()

'''
    Fixed the energy at 0.4kW
'''

# fixed_0.4kW
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 400)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_1.append(1 / rewards)
    avg_rewards_fixed_1.append(np.mean(rewards_list_fixed_1[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_1.append(t)
    avg_rewards_time_list_fixed_1.append(np.mean(rewards_time_list_fixed_1[:]))
    rewards_bak_list_fixed_1.append(bak)
    avg_rewards_bak_list_fixed_1.append(np.mean(rewards_bak_list_fixed_1[:]))
    rewards_bat_list_fixed_1.append(bat)
    avg_rewards_bat_list_fixed_1.append(np.mean(rewards_bat_list_fixed_1[:]))
    avg_rewards_energy_list_fixed_1.append(
        avg_rewards_bak_list_fixed_1[-1]+avg_rewards_bat_list_fixed_1[-1])
    fixed_1_data.append([avg_rewards_time_list_fixed_1[-1],
                        avg_rewards_bak_list_fixed_1[-1], avg_rewards_bat_list_fixed_1[-1]])
    if dones:
        env.reset()

'''
    Fixed the energy at 1kW
'''

# fixed_1kW
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = env.env_method('fixed_action_cal', 1000)
    obs, rewards, dones, info = env.step(action)
    rewards_list_fixed_2.append(1 / rewards)
    avg_rewards_fixed_2.append(np.mean(rewards_list_fixed_2[:]))
    t, bak, bat = env.render()
    rewards_time_list_fixed_2.append(t)
    avg_rewards_time_list_fixed_2.append(np.mean(rewards_time_list_fixed_2[:]))
    rewards_bak_list_fixed_2.append(bak)
    avg_rewards_bak_list_fixed_2.append(np.mean(rewards_bak_list_fixed_2[:]))
    rewards_bat_list_fixed_2.append(bat)
    avg_rewards_bat_list_fixed_2.append(np.mean(rewards_bat_list_fixed_2[:]))
    avg_rewards_energy_list_fixed_2.append(
        avg_rewards_bak_list_fixed_2[-1]+avg_rewards_bat_list_fixed_2[-1])
    fixed_2_data.append([avg_rewards_time_list_fixed_2[-1],
                        avg_rewards_bak_list_fixed_2[-1], avg_rewards_bat_list_fixed_2[-1]])
    if dones:
        env.reset()

'''
    Pick a random energy value.
'''

# random
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list_random.append(1 / rewards)
    avg_rewards_random.append(np.mean(rewards_list_random[:]))
    t, bak, bat = env.render()
    rewards_time_list_random.append(t)
    avg_rewards_time_list_random.append(np.mean(rewards_time_list_random[:]))
    rewards_bak_list_random.append(bak)
    avg_rewards_bak_list_random.append(np.mean(rewards_bak_list_random[:]))
    rewards_bat_list_random.append(bat)
    avg_rewards_bat_list_random.append(np.mean(rewards_bat_list_random[:]))
    avg_rewards_energy_list_random.append(
        avg_rewards_bak_list_random[-1]+avg_rewards_bat_list_random[-1])
    random_data.append([avg_rewards_time_list_random[-1],
                       avg_rewards_bak_list_random[-1], avg_rewards_bat_list_random[-1]])
    if dones:
        env.reset()

'''
    The algorithm we used in the paper, PPO. 
    * Here we simply invoke the PPO module that is already implemented.
    * The informations about this module can be found here: https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
'''

# ppo
set_seed(rand_seed)
obs = env.reset()
for i in range(t_range):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    rewards_list_ppo.append(1 / rewards)
    avg_rewards_ppo.append(np.mean(rewards_list_ppo[:]))
    t, bak, bat = env.render()
    rewards_time_list_ppo.append(t)
    avg_rewards_time_list_ppo.append(np.mean(rewards_time_list_ppo[:]))
    rewards_bak_list_ppo.append(bak)
    avg_rewards_bak_list_ppo.append(np.mean(rewards_bak_list_ppo[:]))
    rewards_bat_list_ppo.append(bat)
    avg_rewards_bat_list_ppo.append(np.mean(rewards_bat_list_ppo[:]))
    avg_rewards_energy_list_ppo.append(
        avg_rewards_bak_list_ppo[-1]+avg_rewards_bat_list_ppo[-1])
    ppo_data.append([avg_rewards_time_list_ppo[-1],
                    avg_rewards_bak_list_ppo[-1], avg_rewards_bat_list_ppo[-1]])
    if dones:
        env.reset()
    # env.render()

'''
    Here we implemented a DQN algorithm to compare with PPO.
    * We use a single hidden layer neural network.
    * The implementation is a stub, tbh.
    
'''

# dqn


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = 1.0

        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=1000000)
        # the neural network
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(
            observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))
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
    # episode = 0
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
            # print('\tstate: ', state)
            if step >= 96:  # Termination of a single episode
                done = True
            solver.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                # episode += 1
                break
            solver.replay()
            # Termination of the entire training process.
            if accumulated_step == train_time_slots:
                terminal = True
                break
        if terminal:
            break

    for _ in range(t_range):
        action = solver.act(state)
        next_state, reward, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, observation_space])
        t, bak, bat = env.render()
        rewards_list_dqn.append(1 / reward)
        avg_rewards_dqn.append(np.mean(rewards_list_dqn[:]))
        rewards_time_list_dqn.append(t)
        avg_rewards_time_list_dqn.append(np.mean(rewards_time_list_dqn[:]))
        rewards_bak_list_dqn.append(bak)
        avg_rewards_bak_list_dqn.append(np.mean(rewards_bak_list_dqn[:]))
        rewards_bat_list_dqn.append(bat)
        avg_rewards_bat_list_dqn.append(np.mean(rewards_bat_list_dqn[:]))
        avg_rewards_energy_list_dqn.append(
            avg_rewards_bak_list_dqn[-1]+avg_rewards_bat_list_dqn[-1])
        dqn_data.append([avg_rewards_time_list_dqn[-1],
                        avg_rewards_bak_list_dqn[-1], avg_rewards_bat_list_dqn[-1]])


agent()

print('--RESULTS--')
print('{:15}{:30}{:10}{:10}{:10}'.format(
    'method', 'total cost', 'time', 'bak', 'bat'))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format(
    'PPO', avg_rewards_ppo[-1], avg_rewards_time_list_ppo[-1], avg_rewards_bak_list_ppo[-1], avg_rewards_bat_list_ppo[-1]))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('random',
      avg_rewards_random[-1], avg_rewards_time_list_random[-1], avg_rewards_bak_list_random[-1], avg_rewards_bat_list_random[-1]))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('myopic',
      avg_rewards_myopic[-1], avg_rewards_time_list_myopic[-1], avg_rewards_bak_list_myopic[-1], avg_rewards_bat_list_myopic[-1]))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('fixed 0.4kW',
      avg_rewards_fixed_1[-1], avg_rewards_time_list_fixed_1[-1], avg_rewards_bak_list_fixed_1[-1], avg_rewards_bat_list_fixed_1[-1]))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('fixed 1kW',
      avg_rewards_fixed_2[-1], avg_rewards_time_list_fixed_2[-1], avg_rewards_bak_list_fixed_2[-1], avg_rewards_bat_list_fixed_2[-1]))
print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format(
    'dqn', avg_rewards_dqn[-1], avg_rewards_time_list_dqn[-1], avg_rewards_bak_list_dqn[-1], avg_rewards_bat_list_dqn[-1]))
end_time = time.time()
print('elapsed time:', end_time-start_time)

# total cost
df = pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_ppo, 'y_2': avg_rewards_random,
                  'y_3': avg_rewards_myopic, 'y_4': avg_rewards_fixed_1, 'y_5': avg_rewards_fixed_2, 'y_6': avg_rewards_dqn})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot('x', 'y_1', data=df, marker='o', markevery=int(
    t_range/10), color='red', linewidth=1, label="ppo")
plt.plot('x', 'y_2', data=df, marker='^', markevery=int(
    t_range/10), color='olive', linewidth=1, label="random")
plt.plot('x', 'y_3', data=df, marker='s', markevery=int(
    t_range/10), color='cyan', linewidth=1, label="myopic")
plt.plot('x', 'y_4', data=df, marker='*', markevery=int(t_range/10),
         color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot('x', 'y_5', data=df, marker='+', markevery=int(t_range/10),
         color='navy', linewidth=1, label="fixed 1kW")
plt.plot('x', 'y_6', data=df, marker='x', markevery=int(
    t_range/10), color='green', linewidth=1, label="Q Learning")
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/avg_total_p='+str(x)+'_.xlsx'
export_excel = df.to_excel(os.path.join(
    my_path, my_file), index=None, header=True)
my_file = 'p='+str(x)+'/avg_total_p='+str(x)+'_.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# time cost
dft = pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_time_list_ppo, 'y_2': avg_rewards_time_list_random, 'y_3': avg_rewards_time_list_myopic,
                   'y_4': avg_rewards_time_list_fixed_1, 'y_5': avg_rewards_time_list_fixed_2, 'y_6': avg_rewards_time_list_dqn})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Delay Cost")
plt.plot('x', 'y_1', data=dft, marker='o', markevery=int(
    t_range/10), color='red', linewidth=1, label="ppo")
plt.plot('x', 'y_2', data=dft, marker='^', markevery=int(
    t_range/10), color='olive', linewidth=1, label="random")
plt.plot('x', 'y_3', data=dft, marker='s', markevery=int(
    t_range/10), color='cyan', linewidth=1, label="myopic")
plt.plot('x', 'y_4', data=dft, marker='*', markevery=int(t_range/10),
         color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot('x', 'y_5', data=dft, marker='+', markevery=int(t_range/10),
         color='navy', linewidth=1, label="fixed 1kW")
plt.plot('x', 'y_6', data=dft, marker='x', markevery=int(
    t_range/10), color='green', linewidth=1, label="Q Learning")
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/avg_time_p='+str(x)+'_.xlsx'
export_excel = dft.to_excel(os.path.join(
    my_path, my_file), index=None, header=True)
my_file = 'p='+str(x)+'/avg_time_p='+str(x)+'_.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# bak-up cost
dfbak = pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_bak_list_ppo, 'y_2': avg_rewards_bak_list_random, 'y_3': avg_rewards_bak_list_myopic,
                     'y_4': avg_rewards_bak_list_fixed_1, 'y_5': avg_rewards_bak_list_fixed_2, 'y_6': avg_rewards_bak_list_dqn})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Back-up Power Cost")
plt.plot('x', 'y_1', data=dfbak, marker='o', markevery=int(
    t_range/10), color='red', linewidth=1, label="ppo")
plt.plot('x', 'y_2', data=dfbak, marker='^', markevery=int(
    t_range/10), color='olive', linewidth=1, label="random")
plt.plot('x', 'y_3', data=dfbak, marker='s', markevery=int(
    t_range/10), color='cyan', linewidth=1, label="myopic")
plt.plot('x', 'y_4', data=dfbak, marker='*', markevery=int(t_range/10),
         color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot('x', 'y_5', data=dfbak, marker='+', markevery=int(t_range/10),
         color='navy', linewidth=1, label="fixed 1kW")
plt.plot('x', 'y_6', data=dfbak, marker='x', markevery=int(
    t_range/10), color='green', linewidth=1, label="Q Learning")
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/avg_backup_p='+str(x)+'_.xlsx'
export_excel = dfbak.to_excel(os.path.join(
    my_path, my_file), index=None, header=True)
my_file = 'p='+str(x)+'/avg_backup_p='+str(x)+'_.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# battery cost
dfbat = pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_bat_list_ppo, 'y_2': avg_rewards_bat_list_random, 'y_3': avg_rewards_bat_list_myopic,
                     'y_4': avg_rewards_bat_list_fixed_1, 'y_5': avg_rewards_bat_list_fixed_2, 'y_6': avg_rewards_bat_list_dqn})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Battery Cost")
plt.plot('x', 'y_1', data=dfbat, marker='o', markevery=int(
    t_range/10), color='red', linewidth=1, label="ppo")
plt.plot('x', 'y_2', data=dfbat, marker='^', markevery=int(
    t_range/10), color='olive', linewidth=1, label="random")
plt.plot('x', 'y_3', data=dfbat, marker='s', markevery=int(
    t_range/10), color='cyan', linewidth=1, label="myopic")
plt.plot('x', 'y_4', data=dfbat, marker='*', markevery=int(t_range/10),
         color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot('x', 'y_5', data=dfbat, marker='+', markevery=int(t_range/10),
         color='navy', linewidth=1, label="fixed 1kW")
plt.plot('x', 'y_6', data=dfbat, marker='x', markevery=int(
    t_range/10), color='green', linewidth=1, label="Q Learning")
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/avg_battery_p='+str(x)+'_.xlsx'
export_excel = dfbat.to_excel(os.path.join(
    my_path, my_file), index=None, header=True)
my_file = 'p='+str(x)+'/avg_battery_p='+str(x)+'_.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# energy cost
dfe = pd.DataFrame({'x': range(t_range), 'y_1': avg_rewards_energy_list_ppo, 'y_2': avg_rewards_energy_list_random, 'y_3': avg_rewards_energy_list_myopic,
                   'y_4': avg_rewards_energy_list_fixed_1, 'y_5': avg_rewards_energy_list_fixed_2, 'y_6': avg_rewards_energy_list_dqn})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Energy Cost")
plt.plot('x', 'y_1', data=dfe, marker='o', markevery=int(
    t_range/10), color='red', linewidth=1, label="ppo")
plt.plot('x', 'y_2', data=dfe, marker='^', markevery=int(
    t_range/10), color='olive', linewidth=1, label="random")
plt.plot('x', 'y_3', data=dfe, marker='s', markevery=int(
    t_range/10), color='cyan', linewidth=1, label="myopic")
plt.plot('x', 'y_4', data=dfe, marker='*', markevery=int(t_range/10),
         color='skyblue', linewidth=1, label="fixed 0.4kW")
plt.plot('x', 'y_5', data=dfe, marker='+', markevery=int(t_range/10),
         color='navy', linewidth=1, label="fixed 1kW")
plt.plot('x', 'y_6', data=dfe, marker='x', markevery=int(
    t_range/10), color='green', linewidth=1, label="Q Learning")
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/avg_energy_p='+str(x)+'_.xlsx'
export_excel = dfe.to_excel(os.path.join(
    my_path, my_file), index=None, header=True)
my_file = 'p='+str(x)+'/avg_energy_p='+str(x)+'_.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()

# ppo area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_ppo,
      avg_rewards_bak_list_ppo, avg_rewards_bat_list_ppo]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('PPO')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/ppo_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# random area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_random,
      avg_rewards_bak_list_random, avg_rewards_bat_list_random]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Random')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/random_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# myopic area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_myopic,
      avg_rewards_bak_list_myopic, avg_rewards_bat_list_myopic]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Myopic')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/myopic_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# fixed 0.4kW area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_fixed_1,
      avg_rewards_bak_list_fixed_1, avg_rewards_bat_list_fixed_1]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Fixed 0.4kW')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/fixed_0.4kW_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# fixed 1kW area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_fixed_2,
      avg_rewards_bak_list_fixed_2, avg_rewards_bat_list_fixed_2]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('Fixed 1kW')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/fixed_1kW_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
# dqn area chart
plt.xlabel("Time Slot")
plt.ylabel("Average Costs")
xx = range(t_range)
yy = [avg_rewards_time_list_dqn,
      avg_rewards_bak_list_dqn, avg_rewards_bat_list_dqn]
fig = plt.stackplot(xx, yy, colors='w', edgecolor='black', labels=[
                    'Delay cost', 'Backup cost', 'Battery cost'])
hatches = ['...', '+++++', '///']
for s, h in zip(fig, hatches):
    s.set_hatch(h)
plt.title('DQN')
plt.legend()
plt.grid()
my_file = 'p='+str(x)+'/dqn_area'+'p='+str(x)+'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()

# df1 = pd.DataFrame(ppo_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df1.plot.area()
# # plt.plot(range(t_range), avg_rewards_ppo)
# plt.grid()
# plt.ylim(0,20)
# plt.title('PPO')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = 'ppo_p='+str(x)+'_.xlsx'
# export_excel = df1.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'ppo_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# df2 = pd.DataFrame(random_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df2.plot.area()
# # plt.plot(range(t_range), avg_rewards_random)
# plt.grid()
# plt.ylim(0,20)
# plt.title('random')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = 'random_p='+str(x)+'_.xlsx'
# export_excel = df2.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'random_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# df3 = pd.DataFrame(myopic_data , columns=['delay cost', 'back-up power cost', 'battery cost'])
# df3.plot.area()
# # plt.plot(range(t_range), avg_rewards_myopic)
# plt.grid()
# plt.ylim(0,20)
# plt.title('myopic')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = 'myopic_p='+str(x)+'_.xlsx'
# export_excel = df3.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'myopic_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# df4 = pd.DataFrame(fixed_1_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df4.plot.area()
# # plt.plot(range(t_range), avg_rewards_fixed_1)
# plt.grid()
# plt.ylim(0,20)
# plt.title('fixed 0.4 kW')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = '04_p='+str(x)+'_.xlsx'
# export_excel = df4.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = '04_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# df5 = pd.DataFrame(fixed_2_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df5.plot.area()
# # plt.plot(range(t_range), avg_rewards_fixed_2)
# plt.grid()
# plt.ylim(0,20)
# plt.title('fixed 1 kW')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = '1_p='+str(x)+'_.xlsx'
# export_excel = df5.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = '1_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()

# df6 = pd.DataFrame(dqn_data, columns=['delay cost', 'back-up power cost', 'battery cost'])
# df6.plot.area()
# # plt.plot(range(t_range), avg_rewards_dqn)
# plt.grid()
# plt.ylim(0,20)
# plt.title('q-learning')
# plt.legend()
# plt.xlabel("Time Slot")
# plt.ylabel("Time Average Cost")
# my_file = 'dqn_p='+str(x)+'_.xlsx'
# export_excel = df6.to_excel (os.path.join(my_path, my_file), index = None, header=True)
# my_file = 'dqn_p='+str(x)+'_.png'
# plt.savefig(os.path.join(my_path, my_file))
# plt.show()
