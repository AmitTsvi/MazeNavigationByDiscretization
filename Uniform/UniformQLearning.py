#!/usr/bin/env python3

from __future__ import print_function

from random import choice
import random
import vizdoom as vzd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from decimal import *
import numpy as np
import datetime
import seaborn as sns
import math

DEFAULT_CONFIG = "../scenarios/my_way_home_allpoints.cfg"
PLOT = True
LOAD = True
SAVE = True


def state_to_bucket(state):
    new_state = (state.game_variables[0], state.game_variables[1], state.game_variables[2])
    # print(new_state)
    x = int((new_state[0]-min_x)/disc_diff)
    y = int((new_state[1]-min_y) / disc_diff)
    # print(tuple([x, y]))
    # return tuple([x, y])
    theta = int((new_state[2]) / disc_angle)
    # print(tuple([x, y, theta]))
    return tuple([x, y, theta])


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = random.randrange(NUM_ACTIONS)
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use information about objects and map.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)

    actions = [[0, True, False, False],
               [90, False, True, False],
               [-90, False, False, True]]
    ticks = 5
    game.init()
    state = game.get_state()
    blocking = []
    for s in state.sectors:
        blocking += [[l.x1, l.y1, l.x2, l.y2] for l in s.lines if l.is_blocking]
    max_x = max([max(b[0], b[2]) for b in blocking])
    min_x = min([min(b[0], b[2]) for b in blocking])
    max_y = max([max(b[1], b[3]) for b in blocking])
    min_y = min([min(b[1], b[3]) for b in blocking])
    print("x range: " + str(min_x) + " " + str(max_x))
    print("y range: " + str(min_y) + " " + str(max_y))
    disc_diff = 20.0

    dummy_state = game.get_state()
    target_object = [o for o in dummy_state.objects if o.name == 'GreenArmor'][0]
    dummy_state.game_variables[0] = target_object.position_x
    dummy_state.game_variables[1] = target_object.position_y

    min_angle = 0
    max_angle = 360
    disc_angle = 90

    plot_every = 100

    # NUM_BUCKETS = (int((max_x-min_x)/disc_diff), int((max_y-min_y)/disc_diff))
    NUM_BUCKETS = (int((max_x-min_x)/disc_diff), int((max_y-min_y)/disc_diff), int((max_angle-min_angle)/disc_angle))
    print("NUM_BUCKETS"+str(NUM_BUCKETS))
    NUM_ACTIONS = len(actions)
    print("NUM_ACTIONS"+str(NUM_ACTIONS))
    # STATE_SPACE = [(x, y) for x in range(NUM_BUCKETS[0]) for y in range(NUM_BUCKETS[1])]
    STATE_SPACE = [(x, y, theta) for x in range(NUM_BUCKETS[0]) for y in range(NUM_BUCKETS[1]) for theta in range(NUM_BUCKETS[2])]

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0

    episodes = 100
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    if LOAD is False:
        q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
        n_visits = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    else:
        f = open('q_function.npy', 'rb')
        q_table = np.load(f)
        f.close()
        f = open('state_visits.npy', 'rb')
        n_visits = np.load(f)
        f.close()

    episode_path = []

    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 1

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        f = open(str(datetime.datetime.now()).split()[0]+'.log', 'a')
        if PLOT and i % plot_every == 0:
            plt.show()
        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        final_state = 0
        total_reward = 0
        first_draw = True
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()
            bucket = state_to_bucket(state)
            episode_path.append(bucket)
            action = select_action(bucket, explore_rate)
            reward = game.make_action(actions[action], ticks)
            # print("reward "+str(reward)+"\n")
            total_reward += reward
            n_visits[bucket + (action,)] += 1

            if not game.is_episode_finished():
                new_state = game.get_state()
                new_bucket = state_to_bucket(new_state)
                best_q = np.amax(q_table[new_bucket])
                q_table[bucket + (action,)] += learning_rate * (
                            reward + discount_factor * (best_q) - q_table[bucket + (action,)])
            elif reward > 0:
                new_bucket = state_to_bucket(dummy_state)
                best_q = np.amax(q_table[new_bucket])
                q_table[bucket + (action,)] += learning_rate * (
                            reward + discount_factor * (best_q) - q_table[bucket + (action,)])

            explore_rate = get_explore_rate(i)
            learning_rate = get_learning_rate(i)

            final_state = state.number

            if PLOT and i % plot_every == 0:
                # Print information about objects present in the episode.
                for o in state.objects:
                    if o.name == "DoomPlayer":
                        plt.plot(o.position_x, o.position_y, color='green', marker='o')
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='o')

                if first_draw is True:
                    for s in state.sectors:
                        for l in s.lines:
                            if l.is_blocking:
                                plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
                    first_draw = False
                # Show map
                # plt.show()
                plt.draw()
                plt.pause(0.001)

        print("Episode finished!")
        f.write("Episode "+str(i)+" State #"+str(final_state)+" Total reward: "+str(total_reward)+"\n")
        f.close()
        if PLOT and i % plot_every == 0:
            plt.savefig("Episode"+str(i))
            plt.close()
            sns.set()
            ax = sns.heatmap(np.transpose(np.sum(n_visits, axis=(2, 3))))
            fig = ax.get_figure()
            fig.savefig("Episode"+str(i)+"_HeatMap")
            plt.close()
        if i == 0:
            plt.show()
            for s in state.sectors:
                for l in s.lines:
                    if l.is_blocking:
                        plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
            plt.xticks([min_x+disc_diff*x for x in range(NUM_BUCKETS[0])], "")
            plt.yticks([min_y+disc_diff*y for y in range(NUM_BUCKETS[1])], "")
            plt.grid(True)
            plt.savefig("Disc_Map")
            plt.close()
    if SAVE is True:
        f = open('q_function.npy', 'wb')
        np.save(f, q_table)
        f.close()
        f = open('state_visits.npy', 'wb')
        np.save(f, n_visits)
        f.close()

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
