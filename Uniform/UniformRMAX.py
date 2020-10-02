#!/usr/bin/env python3

from __future__ import print_function

from random import choice
import random as rnd
import vizdoom as vzd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from decimal import *
import numpy as np
import datetime
import seaborn as sns

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


def select_action(state):
    # print("state" + str(state))
    if np.max(q_table[state]) == np.min(q_table[state]):
        action = rnd.randint(0, len(actions)-1)
    else:
        action = int(np.argmax(q_table[state]))
    return action


def update_value_functions(full=False, path=[]):
    print("@@@@@@@@@@@@@@@@@")
    update = True
    counter = 0
    rnd.shuffle(STATE_SPACE)
    path.reverse()
    while update is True and counter <= 100:
        print("Value iteration iteration #"+str(counter))
        counter += 1
        update = False
        for state in path+STATE_SPACE:
            temp_max = -np.inf
            for action in range(NUM_ACTIONS):
                temp = r_table[state+(action,)]
                non_zero_next = np.transpose(np.nonzero(p_table[state+(action,)]))
                # if non_zero_next.size != 0:
                    # print("Current state= " + str(state))
                    # print(non_zero_next)
                temp += sum([p_table[state+(action,)+tuple(next_state)]*value_table_temp[tuple(next_state)] for next_state in non_zero_next])
                temp_max = max(temp_max, temp)
                if n_visits[state+(action,)] < M_VISITS_TO_KNOWN:
                    q_table[state + (action,)] = R_MAX
                else:
                    q_table[state+(action,)] = temp
            if round(value_table_temp[state], 4) != round(temp_max, 4):
                # print("crr_max= "+str(value_table_temp[state])+" new_max= "+str(temp_max))
                value_table_temp[state] = temp_max
                if full is True:
                    # print("State that cause another iteration: "+str(state))
                    update = True


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

    min_angle = 0
    max_angle = 360
    disc_angle = 90

    update_area = 10

    plot_after = 96

    # NUM_BUCKETS = (int((max_x-min_x)/disc_diff), int((max_y-min_y)/disc_diff))
    NUM_BUCKETS = (int((max_x-min_x)/disc_diff), int((max_y-min_y)/disc_diff), int((max_angle-min_angle)/disc_angle))
    print("NUM_BUCKETS"+str(NUM_BUCKETS))
    NUM_ACTIONS = len(actions)
    print("NUM_ACTIONS"+str(NUM_ACTIONS))
    # STATE_SPACE = [(x, y) for x in range(NUM_BUCKETS[0]) for y in range(NUM_BUCKETS[1])]
    STATE_SPACE = [(x, y, theta) for x in range(NUM_BUCKETS[0]) for y in range(NUM_BUCKETS[1]) for theta in range(NUM_BUCKETS[2])]

    R_MAX = 1.0
    M_VISITS_TO_KNOWN = 1  # Increase for more exploration

    episodes = 100
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    if LOAD is False:
        r_table = np.full(NUM_BUCKETS + (NUM_ACTIONS,), R_MAX, dtype=float)
        p_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,) + NUM_BUCKETS, dtype=float)

        # for state in STATE_SPACE:
        #     for action in range(NUM_ACTIONS):
        #         p_table[state + (action,) + state] = 0.2

        n_visits = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
        n_transitions = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,) + NUM_BUCKETS, dtype=float)

        q_table = np.full(NUM_BUCKETS + (NUM_ACTIONS,), 2*R_MAX, dtype=float)
        value_table_temp = np.zeros(NUM_BUCKETS, dtype=float)
    else:
        f = open('prob.npy', 'rb')
        p_table = np.load(f)
        f.close()
        f = open('value.npy', 'rb')
        value_table_temp = np.load(f)
        f.close()
        f = open('q_function.npy', 'rb')
        q_table = np.load(f)
        f.close()
        f = open('reward.npy', 'rb')
        r_table = np.load(f)
        f.close()
        f = open('state_visits.npy', 'rb')
        n_visits = np.load(f)
        f.close()
        f = open('transitions.npy', 'rb')
        n_transitions = np.load(f)
        f.close()

    episode_path = []

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        f = open(str(datetime.datetime.now()).split()[0]+'.log', 'a')
        if PLOT and i >= plot_after:
            plt.show()
        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        state = game.get_state()
        final_state = 0
        total_reward = 0
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()
            bucket = state_to_bucket(state)
            episode_path.append(bucket)
            action = select_action(bucket)
            reward = game.make_action(actions[action], ticks)
            # print("reward "+str(reward)+"\n")
            total_reward += reward

            n_visits[bucket + (action,)] += 1
            if not game.is_episode_finished():
                new_state = game.get_state()
                new_bucket = state_to_bucket(new_state)
                n_transitions[bucket+(action,)+new_bucket] += 1
            if n_visits[bucket + (action,)] >= M_VISITS_TO_KNOWN:
                r_table[bucket + (action,)] = reward
                for next_bucket in STATE_SPACE:
                    p_table[bucket + (action,) + next_bucket] = n_transitions[bucket + (action,) + next_bucket] / \
                                                                n_visits[bucket + (action,)]
            if state.number % 30 == 0:
                update_value_functions(False, episode_path)

            # print("State #" + str(state.number))
            final_state = state.number
            # print("Player position: x:", state.game_variables[0], ", y:", state.game_variables[1], ", angle:",
            #       state.game_variables[2])
            # print("Objects:")

            if PLOT and i >= plot_after:
                # Print information about objects present in the episode.
                for o in state.objects:
                    if o.name == "DoomPlayer":
                        plt.plot(o.position_x, o.position_y, color='green', marker='o')
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='o')

                # print("=====================")
                # print("Sectors:")
                # Print information about sectors.
                for s in state.sectors:
                    # print("Sector floor height:", s.floor_height, ", ceiling height:", s.ceiling_height)
                    # print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines])

                    # Plot sector on map
                    for l in s.lines:
                        if l.is_blocking:
                            plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

                # Show map
                # plt.show()
                plt.draw()
                plt.pause(0.001)

        print("Episode finished!")
        update_value_functions(True, episode_path)
        f.write("Episode "+str(i)+" State #"+str(final_state)+" Total reward: "+str(total_reward)+"\n")
        f.close()
        if PLOT and i >= plot_after:
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
            # plt.axes.xaxis.set_ticklabels([])
            # plt.axes.yaxis.set_ticklabels([])
            plt.grid(True)
            plt.savefig("Disc_Map")
            plt.close()
    if SAVE is True:
        f = open('prob.npy', 'wb')
        np.save(f, p_table)
        f.close()
        f = open('value.npy', 'wb')
        np.save(f, value_table_temp)
        f.close()
        f = open('q_function.npy', 'wb')
        np.save(f, q_table)
        f.close()
        f = open('reward.npy', 'wb')
        np.save(f, r_table)
        f.close()
        f = open('state_visits.npy', 'wb')
        np.save(f, n_visits)
        f.close()
        f = open('transitions.npy', 'wb')
        np.save(f, n_transitions)
        f.close()
    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
