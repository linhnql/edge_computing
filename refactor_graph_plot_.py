import matplotlib.pyplot as plt
import pandas as pd
import os

my_res_path = os.path.abspath('wcci/')
my_xls_path = os.path.abspath('res/')

val = 0.5  # value of p


def cost(filename, ylabel):
    xls_file = 'p='+str(val)+'/' + filename + '_p='+str(val)+'_.xlsx'

    try:
        plot_list = [[] for _ in range(5)]
        i = 0
        rng = 0
        with open(os.path.join(my_xls_path, xls_file), 'rb') as f:
            xls_data = pd.read_excel(f)
            for i in range(5):
                plot_list[i] = xls_data['y_' + str(i + 1)].tolist()[:1000]
                # print(plot_list[i])
                rng = len(plot_list[i])
                # rng =
            i += 1
        df = pd.DataFrame(
            {'x': range(rng), 'y_1': plot_list[0], 'y_2': plot_list[1], 'y_3': plot_list[2], 'y_4': plot_list[3],
             'y_5': plot_list[4]})  # , 'y_6': plot_list[5]})
        plt.xlabel("Time Slot")
        plt.ylabel(ylabel)
        plt.plot('x', 'y_1', data=df, marker='o', markevery=int(
            rng / 10), color='red', linewidth=1, label="PPO")
        plt.plot('x', 'y_2', data=df, marker='^', markevery=int(
            rng / 10), color='olive', linewidth=1, label="Random")
        plt.plot('x', 'y_3', data=df, marker='s', markevery=int(
            rng / 10), color='cyan', linewidth=1, label="Myopic")
        plt.plot('x', 'y_4', data=df, marker='*', markevery=int(rng / 10), color='skyblue', linewidth=1,
                 label="Fixed 0.4kW")
        plt.plot('x', 'y_5', data=df, marker='+', markevery=int(rng / 10),
                 color='navy', linewidth=1, label="Fixed 1kW")
        # plt.plot('x', 'y_6', data=df, marker='x', markevery=int(
        #     rng / 10), color='green', linewidth=1, label="Q Learning")
        plt.legend(fancybox=True, shadow=True)
        plt.grid()
        fig_file = 'p=' + str(val) + '/' + filename + \
            '_p=' + str(val) + '_.png'
        plt.savefig(os.path.join(my_res_path, fig_file))
        plt.show()
    except IOError:
        print('Try again')


# # total cost
cost('avg_total', "Time Average Cost")

# # time cost
cost('avg_time', 'Time Average Delay Cost')

# # back-up cost
cost('avg_backup', 'Time Average Back-up Power Cost')

# # battery cost
cost('avg_battery', 'Time Average Battery Power Cost')

# # energy cost
cost('avg_energy', 'Time Average Energy Cost')

avg_time = 'p='+str(val)+'/avg_time_p='+str(val)+'_.xlsx'
avg_backup = 'p='+str(val)+'/avg_backup_p='+str(val)+'_.xlsx'
avg_battery = 'p='+str(val)+'/avg_battery_p='+str(val)+'_.xlsx'

xls_file_names = [avg_time, avg_backup, avg_battery]


def area_chart(title, method, y_col, ylim=0):
    plot_list = [[] for _ in range(3)]
    i = 0
    rng = 0
    for name in xls_file_names:
        try:
            with open(os.path.join(my_xls_path, name), 'rb') as f:
                xls_data = pd.read_excel(f)
                plot_list[i] = xls_data[y_col].tolist()[:1000]
                rng = len(plot_list[i])
                i += 1
        except IOError:
            print('Try again')

    xx = range(rng)
    fig = plt.stackplot(xx, plot_list, edgecolor='black',
                        labels=['Delay cost', 'Backup cost', 'Battery cost'])
    hatches = ['...', '+++++', '///']
    for s, h in zip(fig, hatches):
        s.set_hatch(h)
    if ylim:
        plt.ylim(0, 12)
    plt.title(title)
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.),
               ncol=3, fancybox=True, shadow=True)
    if ylim > 1:
        plt.ylim(0, 12)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    my_file = 'p='+str(val)+'/' + method + '_area_p='+str(val)+'.png'
    plt.savefig(os.path.join(my_res_path, my_file))
    plt.show()


# area chart
area_chart('PPO', 'ppo', 'y_1', 1)
area_chart('Random', 'random', 'y_2')
area_chart('Myopic', 'myopic', 'y_3', 2)
area_chart('Fixed 0.4kW', 'fixed_0.4kW', 'y_4')
area_chart('Fixed 1kW', 'fixed_1kW', 'y_5')
area_chart('DQN', 'dqn', 'y_6', 1)
