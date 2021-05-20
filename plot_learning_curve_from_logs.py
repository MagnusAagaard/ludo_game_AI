import matplotlib.pyplot as plt
import numpy as np

def get_n_gens(log):
    gen = 0
    for line in log:
        pos = line.find('Gen')
        if pos != -1:
            gen = int(line[line.find('Gen') + 4:line.find(',', pos + 4)])
    return gen + 1

def get_pop_size(log):
    size = 0
    for line in log:
        size += 1
        if line.find('Gen') != -1:
            return size - 1

def get_min_max_avg(log, pop_size):
    mins = []
    maxs = []
    avgs = []
    total_won = 0
    minimum = 100
    maximum = 0
    for line in log:
        if line.find('Gen') != -1:
            avgs.append(total_won/pop_size)
            mins.append(minimum)
            maxs.append(maximum)
            total_won = 0
            minimum = 100
            maximum = 0
        else:
            pos = line.find('won')
            if pos != -1:
                val = int(line[pos + 4:line.find(',', pos + 4)])
                total_won += val
                if val < minimum:
                    minimum = val
                if val > maximum:
                    maximum = val
            else:
                print('Could not find games won..')
    return np.array(mins), np.array(maxs), np.array(avgs)

def plot_values(minimums, maximums, averages, name):
    plt.figure(1, dpi=150)
    avg_line,_,errorlines = plt.errorbar(np.arange(len(averages)), averages, yerr=[averages-minimums, maximums-averages], capsize=3, ecolor='0.6')
    # change errorlines to dashes
    errorlines[0].set_linestyle('--')
    # change the maximum generation values errorline to red color
    max_index = np.argmax(maximums)
    print('Maximum value: {}, at generation: {}'.format(np.max(maximums), max_index))
    colors = []
    for i in range(len(errorlines[0].get_segments())):
        if i == max_index:
            colors.append(np.array([1.0, 0, 0, 1]))
        else:
            colors.append(errorlines[0].get_colors()[0])
    errorlines[0].set_color(colors)
    # plot settings
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Learning Curve')
    avg_line.set_label('Average')
    plt.legend(loc='lower right')
    # save plot
    plt.savefig('./figures/' + name[:-4] + '.eps')

def main():
    name = 'run1_wr85_attack.log'
    log_file = './logs/' + name
    lines = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    n_gen = get_n_gens(lines)
    pop_size = get_pop_size(lines)
    minimums, maximums, averages = get_min_max_avg(lines, pop_size)
    plot_values(minimums, maximums, averages, name)

if __name__ == "__main__":
    main()