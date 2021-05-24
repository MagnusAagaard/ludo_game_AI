import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_values(data, name):
    #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'])
    #print(mpl.rcParams['axes.prop_cycle'])
    plt.figure(1, dpi=150)
    plt.plot(data)
    plt.xlabel('Number of games')
    plt.ylabel('Win rate [%]')
    plt.ylim(0,100)
    plt.title('Win rates vs random players')
    plt.legend(['GA Agent', 'Player 2', 'Player 3', 'Player 4'], loc='upper right')
    plt.savefig('./figures/' + name + '.eps')
    plt.show()


def main():
    name = 'eval_winrates_random.log'
    log_file = './logs/' + name
    lines = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    lines = lines[3+100:]
    data_array = []
    for line in lines:
        arr = [float(val) for val in line[1:-2].split(',')]
        data_array.append(arr)
    print(data_array[-1])
    plot_values(data_array, name[14:name.find('.')])
    

if __name__ == "__main__":
    main()