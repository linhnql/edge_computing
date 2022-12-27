import matplotlib.pyplot as plt
import gym_offload_autoscale
import pandas as pd
import os

my_path = os.path.abspath('res/')

# rand_seed = 1234
x = 0.5

# train_time_slots = 20000
t_range = 2000


def result(alg_name, avg_rewards, avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list):
    print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format(alg_name, avg_rewards[-1], avg_rewards_time_list[-1],
                                                       avg_rewards_bak_list[-1], avg_rewards_bat_list[-1]))


def area_chart(alg_name, avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list):
    xx = range(t_range)
    yy = [avg_rewards_time_list, avg_rewards_bak_list, avg_rewards_bat_list]
    # color_map = ["c", "m", "#34495e", "#2ecc71"]  # 'w'-none // ['blue', 'orange','brown']
    fig = plt.stackplot(xx, yy, edgecolor='black', labels=['Delay cost', 'Backup cost', 'Battery cost'])
    hatches = ['...', '+++++', '///']
    for s, h in zip(fig, hatches):
        s.set_hatch(h)
    plt.title(alg_name)
    plt.xlabel("Time Slot")
    plt.ylabel("Average Costs")
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.),
    #         ncol=3, fancybox=True, shadow=True)
    plt.legend()
    plt.grid()

    my_file = 'p='+str(x)+'/'+alg_name+'_area_p='+str(x)+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    plt.show()


def cost(file_name, cost_name, y1, y2, y3, y4, y5):  # , y6):

    df = pd.DataFrame({'x': range(t_range), 'y_1': y1, 'y_2': y2,
                      'y_3': y3, 'y_4': y4, 'y_5': y5})  # , 'y_6': y6})

    plt.xlabel("Time Slot")
    plt.ylabel(cost_name)
    plt.plot('x', 'y_1', data=df, marker='o', markevery=int(t_range/10),
             color='red', linewidth=1, label="PPO")
    plt.plot('x', 'y_2', data=df, marker='^', markevery=int(t_range/10),
             color='olive', linewidth=1, label="Random")
    plt.plot('x', 'y_3', data=df, marker='s', markevery=int(t_range/10),
             color='cyan', linewidth=1, label="Myopic")
    plt.plot('x', 'y_4', data=df, marker='*', markevery=int(t_range/10),
             color='skyblue', linewidth=1, label="Fixed 0.4kW")
    plt.plot('x', 'y_5', data=df, marker='+', markevery=int(t_range/10),
             color='navy', linewidth=1, label="Fixed 1kW")
    # plt.plot('x', 'y_6', data=df, marker='x', markevery=int(t_range/10), color='green', linewidth=1, label="Q Learning")
    plt.legend()
    plt.grid()

    file_xlsx = 'p=' + str(x) + '/' + file_name + '_p=' + str(x) + '_.xlsx'
    export_excel = df.to_excel(os.path.join(my_path, file_xlsx), index=None, header=True)
    file_png = 'p=' + str(x) + '/' + file_name + '_p=' + str(x) + '_.png'
    plt.savefig(os.path.join(my_path, file_png))
    plt.show()
