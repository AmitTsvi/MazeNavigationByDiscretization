import numpy as np
from adaptive_model_Agent_Multiple import AdaptiveModelBasedDiscretization
from src import agent
import vizdoom as vzd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import pickle
import sys, select


DEFAULT_CONFIG = "../GeneralFiles/my_way_home_onespawn.cfg"
PLOT = True
LOAD = False
SAVE = True
TARGET_REWARD = 0.0


def check_for_rescale(state):
    x_width = max_x-min_x
    y_width = max_y-min_y
    new_min_x = min_x - x_width
    new_max_x = max_x + x_width
    new_min_y = min_y - y_width
    new_max_y = max_y + y_width
    # currently always rescaling by doubling on each axis
    # numbering according to graph quadrants order
    if (state.game_variables[0] < min_x and state.game_variables[1] < min_y) or \
            (state.game_variables[0] < min_x and state.game_variables[1] <= max_y):  # only x is out of border
        agent.rescale(1)  # The current scope will be Quadrant I of the new scope
        return tuple([max_x, new_min_x, max_y, new_min_y])
    if (state.game_variables[0] > max_x and state.game_variables[1] < min_y) or\
            (state.game_variables[0] >= min_x and state.game_variables[1] < min_y):  # only y is out of border
        agent.rescale(2)  # The current scope will be Quadrant II of the new scope
        return tuple([new_max_x, min_x, max_y, new_min_y])
    if (state.game_variables[0] > max_x and state.game_variables[1] > max_y) or \
            (state.game_variables[0] > max_x and state.game_variables[1] >= min_y):  # only x is out of border
        agent.rescale(3)  # The current scope will be Quadrant II of the new scope
        return tuple([new_max_x, min_x, new_max_y, min_y])
    if (state.game_variables[0] < min_x and state.game_variables[1] > max_y) or \
            (state.game_variables[0] <= max_x and state.game_variables[1] > max_y):  # only y is out of border
        agent.rescale(4)  # The current scope will be Quadrant II of the new scope
        return tuple([max_x, new_min_x, new_max_y, min_y])
    return tuple([max_x, min_x, max_y, min_y])
    # assign to x_max x_min...
    # find the scale in which to rescale (double won't always be enough?


def add_obs_to_heat_map(state, action):
    new_state = (state.game_variables[0], state.game_variables[1], state.game_variables[2])
    x = int((new_state[0]-min_x_heatmap) / disc_diff)
    y = int((new_state[1]-min_y_heatmap) / disc_diff)
    theta = int((new_state[2]) / disc_angle)
    bucket = tuple([x, y, theta])
    n_visits[bucket + (action,)] += 1


def norm_x(x):
    return (x-min_x)/(max_x-min_x)


def norm_y(y):
    return (y-min_y)/(max_y-min_y)


def select_action(state, h):
    raw_action, active_node = agent.pick_action(state, h)
    num_actions = len(actions)
    action = int(num_actions*raw_action)
    if action == 3:  # in case the raw action is 1.0
        action = 2
    return action, active_node, raw_action


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
    nEps = 3000
    scaling = 0
    alpha = 0
    plot_every = 1
    R_MAX = 1.0
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
    game.set_window_visible(False)
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

    # Initial borders
    initial_width = 20
    initial_height = 20
    dummy_state = game.get_state()
    min_x = dummy_state.game_variables[0] - initial_width/2
    max_x = dummy_state.game_variables[0] + initial_width/2
    min_y = dummy_state.game_variables[1] - initial_height/2
    max_y = dummy_state.game_variables[1] + initial_height/2

    # Find coordinates of target to create the terminal state
    target_object = [o for o in dummy_state.objects if o.name == 'GreenArmor'][0]
    dummy_state.game_variables[0] = target_object.position_x
    dummy_state.game_variables[1] = target_object.position_y

    # For Heat Map
    max_x_heatmap, min_x_heatmap, max_y_heatmap, min_y_heatmap = find_maze_borders(dummy_state)
    disc_diff = 20.0
    min_angle = 0
    max_angle = 360
    disc_angle = 90
    NUM_ACTIONS = len(actions)
    NUM_BUCKETS = (int((max_x_heatmap-min_x_heatmap)/disc_diff), int((max_y_heatmap-min_y_heatmap)/disc_diff),
                   int((max_angle-min_angle)/disc_angle))

    if LOAD is True:
        infile = open('PickledAgent', 'rb')
        agent = pickle.load(infile)
        infile.close()
        f = open('state_visits.npy', 'rb')
        n_visits = np.load(f)
        f.close()
    else:
        agent = AdaptiveModelBasedDiscretization(1, True, R_MAX, NUM_ACTIONS)
        n_visits = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    for i in range(nEps):
        print("Episode #" + str(i + 1)+". 2 seconds to end run")
        # q, o, e = select.select([sys.stdin], [], [], 2)  # TODO: uncomment in linux
        # if (q):
        #     outfile = open("PickledAgent", 'wb')
        #     pickle.dump(agent, outfile)
        #     outfile.close()
        #     f = open('state_visits.npy', 'wb')
        #     np.save(f, n_visits)
        #     f.close()
        #     game.close()
        #     exit()
        f = open(str(datetime.datetime.now()).split()[0]+'.log', 'a')
        if PLOT and i % plot_every == 0:
            plt.show()
        game.new_episode()
        final_state = 0
        total_reward = 0
        t = 0
        while not game.is_episode_finished():
            state = game.get_state()
            bucket = state_to_bucket(state)
            action, active_node, raw_action = select_action(bucket, 0)
            reward = game.make_action(actions[action], ticks)
            total_reward += reward
            add_obs_to_heat_map(state, action)
            if game.is_episode_finished() and reward > TARGET_REWARD:
                max_x, min_x, max_y, min_y = check_for_rescale(dummy_state)
                new_bucket = state_to_bucket(dummy_state)
                agent.update_obs(bucket, raw_action, reward, new_bucket, 0, active_node)
            elif not game.is_episode_finished():
                new_state = game.get_state()
                max_x, min_x, max_y, min_y = check_for_rescale(new_state)
                new_bucket = state_to_bucket(new_state)
                agent.update_obs(bucket, raw_action, reward, new_bucket, 0, active_node)
            if t % 20 == 0:
                agent.update_policy(i)
            t = t + 1

            print("State #" + str(state.number))
            final_state = state.number

            if PLOT and i % plot_every == 0:
                # Print information about objects present in the episode.
                for o in state.objects:
                    if o.name == "DoomPlayer":
                        plt.plot(o.position_x, o.position_y, color='green', marker='o')
                    else:
                        plt.plot(o.position_x, o.position_y, color='red', marker='o')

                if t == 1:
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

        if PLOT and i % plot_every == 0:
            fig = plt.figure(dpi=900)
            tree = agent.tree_list[0]
            tree.plot(fig)
            ax = plt.gca()
            for s in dummy_state.sectors:
                for l in s.lines:
                    if l.is_blocking:
                        ax.plot([norm_x(l.x1), norm_x(l.x2)], [norm_y(l.y1), norm_y(l.y2)], color='black', linewidth=2)
            fig.savefig('AdaptiveDiscretization_Episode#'+str(i)+'_Tree#'+str(0), dpi=900)
            plt.close()

        if PLOT and i % plot_every == 0:
            sns.set()
            ax = sns.heatmap(np.transpose(np.sum(n_visits, axis=(2, 3))))
            ax.invert_yaxis()
            fig = ax.get_figure()
            fig.savefig("Episode#"+str(i)+"_HeatMap")
            plt.close()

    if SAVE is True:
        outfile = open("PickledAgent", 'wb')
        pickle.dump(agent, outfile)
        outfile.close()
        f = open('state_visits.npy', 'wb')
        np.save(f, n_visits)
        f.close()
    game.close()
