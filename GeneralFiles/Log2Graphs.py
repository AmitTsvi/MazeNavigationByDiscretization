import matplotlib.pyplot as plt
import numpy as np

PATH = "C:/Users/amitt/Desktop/TestLog/"
FILES = ['2020-10-05.log', '2020-10-06.log', '2020-10-06_2.log']


def sort_by_episode(a, i):
    return a[i]


if __name__ == "__main__":
    # Get data
    steps = []
    total_reward = []
    for file in FILES:
        f = open(PATH+file, 'r')
        for line in f:
            split_line = line.split(" ")
            steps.append(int(split_line[3].split('#')[1]))
            total_reward.append(round(float(split_line[6].split('\n')[0]), 3))
        f.close()
    # Plot steps graph
    plt.figure()
    plt.plot(steps, 'r.')
    plt.xlabel('Episode #')
    plt.ylabel('Steps to completion')
    plt.savefig(PATH+'StepsVSEpisode', bbox_inches='tight')
    # Plot reward graph
    plt.figure()
    plt.plot(total_reward, 'b.')
    plt.xlabel('Episode #')
    plt.ylabel('Total reward')
    plt.savefig(PATH+'TotalRewardVSEpisode', bbox_inches='tight')
    # Plot accumulated reward
    acc_reward = [sum(total_reward[:i]) for i in range(len(total_reward))]
    plt.figure()
    plt.plot(acc_reward, 'g.')
    plt.xlabel('Episode #')
    plt.ylabel('Accumulated reward')
    plt.savefig(PATH+'AccumulatedRewardVSEpisode', bbox_inches='tight')
