import matplotlib.pyplot as plt
import numpy as np

def plot_values(data, name):
    plt.figure(1, dpi=150)
    plt.plot(data)
    plt.xlabel('Number of games')
    plt.ylabel('Win rate [%]')
    plt.title('Win rates vs random players')
    plt.legend(['Agent', 'Player 2', 'Player 3', 'Player 4'], loc='right')
    plt.savefig('./figures/' + name + '.eps')
    plt.show()


def main():
    name = 'eval_winrates_scaled_attack.log'
    log_file = './logs/' + name
    lines = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    lines = lines[3+10:]
    data_array = []
    for line in lines:
        arr = [float(val) for val in line[1:-2].split(',')]
        data_array.append(arr)
    
    plot_values(data_array, name[14:name.find('.')])
    

if __name__ == "__main__":
    main()