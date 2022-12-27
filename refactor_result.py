import gym_offload_autoscale

import refactor_ppo as ppo
import refactor_random as random
import refactor_myopic as myopic
import refactor_fixed_04 as fixed_04
import refactor_fixed_1 as fixed_1
# import refactor_dqn as dqn

from refactor_func import *


print('--RESULTS--')
print('{:15}{:30}{:10}{:10}{:10}'.format('method', 'total cost', 'time', 'bak', 'bat'))
result('PPO', ppo.avg_rewards, ppo.avg_rewards_time_list,
       ppo.avg_rewards_bak_list, ppo.avg_rewards_bat_list)
result('Random', random.avg_rewards, random.avg_rewards_time_list,
       random.avg_rewards_bak_list, random.avg_rewards_bat_list)
result('Myopic', myopic.avg_rewards, myopic.avg_rewards_time_list,
       myopic.avg_rewards_bak_list, myopic.avg_rewards_bat_list)
result('Fixed 0.4kW', fixed_04.avg_rewards, fixed_04.avg_rewards_time_list,
       fixed_04.avg_rewards_bak_list, fixed_04.avg_rewards_bat_list)
result('Fixed 1kW', fixed_1.avg_rewards, fixed_1.avg_rewards_time_list,
       fixed_1.avg_rewards_bak_list, fixed_1.avg_rewards_bat_list)
# result('DQN', alg_dqn.avg_rewards, alg_dqn.avg_rewards_time_list,
#        alg_dqn.avg_rewards_bak_list, alg_dqn.avg_rewards_bat_list)


cost("avg_total", "Time Average Cost", ppo.avg_rewards, random.avg_rewards,
     myopic.avg_rewards, fixed_04.avg_rewards, fixed_1.avg_rewards)  # , alg_dqn.avg_rewards)

cost("avg_time", "Time Average Delay Cost", ppo.avg_rewards_time_list,
     random.avg_rewards_time_list, myopic.avg_rewards_time_list,
     fixed_04.avg_rewards_time_list, fixed_1.avg_rewards_time_list)  # , alg_dqn.avg_rewards_time_list)

cost("avg_backup", "Time Average Back-up Power Cost", ppo.avg_rewards_bak_list,
     random.avg_rewards_bak_list, myopic.avg_rewards_bak_list,
     fixed_04.avg_rewards_bak_list, fixed_1.avg_rewards_bak_list)  # , alg_dqn.avg_rewards_bak_list)

cost("avg_battery", "Time Average Battery Cost", ppo.avg_rewards_bat_list,
     random.avg_rewards_bat_list, myopic.avg_rewards_bat_list, fixed_04.avg_rewards_bat_list,
     fixed_1.avg_rewards_bat_list)  # , alg_dqn.avg_rewards_bat_list)

cost("avg_energy", "Time Average Energy Cost", ppo.avg_rewards_energy_list,
     random.avg_rewards_energy_list, myopic.avg_rewards_energy_list,
     fixed_04.avg_rewards_energy_list, fixed_1.avg_rewards_energy_list)  # , alg_dqn.avg_rewards_energy_list)


area_chart('PPO', ppo.avg_rewards_time_list,
           ppo.avg_rewards_bak_list, ppo.avg_rewards_bat_list)
area_chart('Random', random.avg_rewards_time_list,
           random.avg_rewards_bak_list, random.avg_rewards_bat_list)
area_chart('Myopic', myopic.avg_rewards_time_list,
           myopic.avg_rewards_bak_list, myopic.avg_rewards_bat_list)
area_chart('Fixed 0.4kW', fixed_04.avg_rewards_time_list,
           fixed_04.avg_rewards_bak_list, fixed_04.avg_rewards_bat_list)
area_chart('Fixed 1kW', fixed_1.avg_rewards_time_list,
           fixed_1.avg_rewards_bak_list, fixed_1.avg_rewards_bat_list)
# area_chart('DQN', dqn.avg_rewards_time_list,
#            dqn.avg_rewards_bak_list, dqn.avg_rewards_bat_list, 1)
