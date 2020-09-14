import numpy as np
from adaptive_model_Agent_Multiple import AdaptiveModelBasedDiscretization
from src import agent
import vizdoom as vzd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import copy

DEFAULT_CONFIG = "../../../scenarios/my_way_home_allpoints.cfg"
PLOT = True
LOAD = False
SAVE = False
TARGET_STATE = tuple([1040.0, -352.0, 0.0])  # TODO: try to get this auto
TARGET_REWARD = 0.0


def select_action(state, h):
    raw_action = agent.pick_action(state, h)
    num_actions = len(actions)
    action = int(num_actions*raw_action)
    if action == 3:  # in case the raw action is 1.0
        action = 2
    return action


def state_to_bucket(state):
    new_state = (state.game_variables[0], state.game_variables[1], state.game_variables[2])
    # print(new_state)
    x = (new_state[0]-min_x)/(max_x-min_x)
    y = (new_state[1]-min_y)/(max_y-min_y)
    theta = new_state[2]/360
    # print(tuple([x, y, theta]))
    return tuple([x, y, theta])


def find_maze_borders(state):
    blocking = []
    for s in state.sectors:
        blocking += [[l.x1, l.y1, l.x2, l.y2] for l in s.lines if l.is_blocking]
    max_x = max([max(b[0], b[2]) for b in blocking])
    min_x = min([min(b[0], b[2]) for b in blocking])
    max_y = max([max(b[1], b[3]) for b in blocking])
    min_y = min([min(b[1], b[3]) for b in blocking])
    print("x range: " + str(min_x) + " " + str(max_x))
    print("y range: " + str(min_y) + " " + str(max_y))
    return tuple([max_x, min_x, max_y, min_y])


if __name__ == "__main__":
    epLen = 420
    nEps = 400
    numIters = 25
    scaling = 0
    alpha = 0
    plot_every = 25
    R_MAX = 1.0
    M_VISITS_TO_KNOWN = 1
    VALUE_ITERATIONS = 1

    # Init env
    parser = ArgumentParser("ViZDoom example showing how to use information about objects and map.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../scenarios/*cfg for more scenarios.")
    args = parser.parse_args()
    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.init()

    actions = [[0, True, False, False],
               [90, False, True, False],
               [-90, False, False, True]]
    ticks = 5

    # Find borders to normalize state space to [0,1]x[0,1]
    dummy_state = game.get_state()
    max_x, min_x, max_y, min_y = find_maze_borders(dummy_state)

    agent = AdaptiveModelBasedDiscretization(epLen, nEps, scaling, alpha, False, 2*R_MAX)  # TODO: RMAX or 2*RMAX

    for i in range(nEps):
        print("Episode #" + str(i + 1))
        f = open(str(datetime.datetime.now()).split()[0]+'.log', 'a')
        if PLOT and i % plot_every == 0:
            plt.show()
        game.new_episode()
        state = game.get_state()
        final_state = 0
        total_reward = 0
        h = 0
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()
            bucket = state_to_bucket(state)
            action = select_action(bucket, h)
            reward = game.make_action(actions[action], ticks)
            total_reward += reward
            if game.is_episode_finished() and reward > TARGET_REWARD:
                dummy_state.game_variables[0] = TARGET_STATE[0]
                dummy_state.game_variables[1] = TARGET_STATE[1]
                dummy_state.game_variables[2] = TARGET_STATE[2]
                new_bucket = state_to_bucket(dummy_state)
                agent.update_obs(bucket, action, reward, new_bucket, h)
            elif not game.is_episode_finished():
                new_state = game.get_state()
                new_bucket = state_to_bucket(new_state)
                agent.update_obs(bucket, action, reward, new_bucket, h)
            h = h + 1

            print("State #" + str(state.number))
            final_state = state.number
            # print("Player position: x:", state.game_variables[0], ", y:", state.game_variables[1], ", angle:",
            #       state.game_variables[2])
            # print("Objects:")

            if PLOT and i % plot_every == 0:
                # Print information about objects present in the episode.
                for o in state.objects:
                    if o.name == "DoomPlayer":
                        plt.plot(o.position_x, o.position_y, color='green', marker='o')
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='o')

                # print("=====================")
                # print("Sectors:")
                for s in state.sectors:
                    # Plot sector on map
                    for l in s.lines:
                        if l.is_blocking:
                            plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
                plt.draw()
                plt.pause(0.001)
        for j in range(VALUE_ITERATIONS):
            agent.update_policy(i)
        print("Episode finished!")
        f.write("Episode "+str(i)+" State #"+str(final_state)+" Total reward: "+str(total_reward)+"\n")
        f.close()
        if PLOT and i % plot_every == 0:
            plt.savefig("Episode" + str(i))
            plt.close()

        # TODO: plot a continuous graph of balls' n_visits
        # if PLOT and i % plot_every == 0:
        #     sns.set()
        #     ax = sns.heatmap(np.transpose(np.sum(n_visits, axis=(2, 3))))
        #     fig = ax.get_figure()
        #     fig.savefig("Episode"+str(i)+"_HeatMap")
        #     plt.close()
        # TODO: plot a graph of the space's division to balls
        # if i == 0:
        #     plt.show()
        #     for s in state.sectors:
        #         for l in s.lines:
        #             if l.is_blocking:
        #                 plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
        #     plt.xticks([min_x+disc_diff*x for x in range(NUM_BUCKETS[0])], "")
        #     plt.yticks([min_y+disc_diff*y for y in range(NUM_BUCKETS[1])], "")
        #     # plt.axes.xaxis.set_ticklabels([])
        #     # plt.axes.yaxis.set_ticklabels([])
        #     plt.grid(True)
        #     plt.savefig("Disc_Map")
        #     plt.close()
    game.close()
