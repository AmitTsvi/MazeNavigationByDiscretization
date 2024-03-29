import numpy as np
from adaptive_model_Agent_Multiple import AdaptiveModelBasedDiscretization
import vizdoom as vzd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import pickle
import sys
import select
import time
import matplotlib as mpl
import config


def check_for_rescale(state):
    x = state.game_variables[0]
    y = state.game_variables[1]
    x_width = max_x-min_x
    y_width = max_y-min_y
    new_min_x = min_x - x_width
    new_max_x = max_x + x_width
    new_min_y = min_y - y_width
    new_max_y = max_y + y_width
    if (min_x <= x <= max_x) and (min_y <= y <= max_y):
        return tuple([max_x, min_x, max_y, min_y])
    # currently always rescaling by doubling on each axis
    # numbering according to graph quadrants order
    if (x < min_x and y <= min_y+0.5*y_width) or (y < min_y and x <= min_x + 0.5 * x_width):
        agent.rescale(1)  # The current scope will be Quadrant I of the new scope
        return tuple([max_x, new_min_x, max_y, new_min_y])
    if (x > max_x and y <= min_y+0.5*y_width) or (y < min_y and x > max_x - 0.5 * x_width):
        agent.rescale(2)  # The current scope will be Quadrant II of the new scope
        return tuple([new_max_x, min_x, max_y, new_min_y])
    if (x > max_x and y > max_y-0.5*y_width) or (y > max_y and x > max_x - 0.5 * x_width):
        agent.rescale(3)  # The current scope will be Quadrant III of the new scope
        return tuple([new_max_x, min_x, new_max_y, min_y])
    if (x < min_x and y > max_y-0.5*y_width) or (y > max_y and x <= min_x + 0.5 * x_width):
        agent.rescale(4)  # The current scope will be Quadrant IV of the new scope
        return tuple([max_x, new_min_x, new_max_y, min_y])


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


def select_action(state):
    raw_action, active_node = agent.pick_action(state)
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


def plot_discretization(i, t):
    disc_name = 'AdaptiveDiscretization_Episode#' + str(i) + '_Step#' + str(t)
    disc_fig = plt.figure(disc_name, dpi=900)
    agent.tree.plot(disc_fig)
    ax = plt.gca()
    for s in dummy_state.sectors:
        for l in s.lines:
            if l.is_blocking:
                ax.plot([norm_x(l.x1), norm_x(l.x2)], [norm_y(l.y1), norm_y(l.y2)], color='black', linewidth=2)
    disc_fig.savefig(disc_name, dpi=900)
    plt.close(disc_name)


def plot_heatmap(i):
    heatmap_name = "Episode#" + str(i) + "_HeatMap"
    sns.set()
    ax = sns.heatmap(np.transpose(np.sum(n_visits, axis=(2, 3))))
    ax.invert_yaxis()
    heatmap_fig = ax.get_figure()
    heatmap_fig.savefig(heatmap_name)
    plt.close()


def plot_path(state):
    plt.figure(path_name)
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


if __name__ == "__main__":
    # Init env
    game = vzd.DoomGame()
    game.load_config(config.DEFAULT_CONFIG)
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
    initial_width = 40
    initial_height = 40
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

    if config.LOAD is True:
        infile = open('PickledAgent', 'rb')
        agent = pickle.load(infile)
        infile.close()
        max_x, min_x, max_y, min_y = agent.get_limits()
        f = open('state_visits.npy', 'rb')
        n_visits = np.load(f)
        f.close()
    else:
        agent = AdaptiveModelBasedDiscretization(NUM_ACTIONS)
        n_visits = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
        agent.set_limits((max_x, min_x, max_y, min_y))

    for i in range(config.NEPS):
        start = time.time()
        print("Episode #" + str(i)+". Press ENTER to end run")
        q, o, e = select.select([sys.stdin], [], [], 2)
        if (q):
            outfile = open("PickledAgent", 'wb')
            pickle.dump(agent, outfile)
            outfile.close()
            f = open('state_visits.npy', 'wb')
            np.save(f, n_visits)
            f.close()
            game.close()
            exit()
        f = open(str(datetime.datetime.now()).split()[0]+'.log', 'a')
        if config.PLOT and i % config.PLOT_EVERY == 0:
            path_name = "Episode" + str(i)
            path_fig = plt.figure(path_name)
        game.new_episode()
        final_state = 0
        total_reward = 0
        t = 0
        if i == 0:
            plot_discretization(i, t)
        while not game.is_episode_finished():
            state = game.get_state()
            bucket = state_to_bucket(state)
            action, active_node, raw_action = select_action(bucket)
            reward = game.make_action(actions[action], ticks)
            total_reward += reward
            add_obs_to_heat_map(state, action)
            old_borders = (max_x, min_x, max_y, min_y)
            if game.is_episode_finished() and reward > 0:
                max_x, min_x, max_y, min_y = check_for_rescale(dummy_state)
                bucket = state_to_bucket(state)  # recalculate bucket due to border changes
                new_bucket = state_to_bucket(dummy_state)
                agent.update_obs(bucket, raw_action, reward, new_bucket, 0, active_node)
            elif not game.is_episode_finished():
                new_state = game.get_state()
                max_x, min_x, max_y, min_y = check_for_rescale(new_state)
                bucket = state_to_bucket(state)  # recalculate bucket due to border changes
                new_bucket = state_to_bucket(new_state)
                agent.update_obs(bucket, raw_action, reward, new_bucket, 0, active_node)
            new_borders = (max_x, min_x, max_y, min_y)
            if t % 20 == 0:
                agent.update_policy()
            if t % 100 == 0:
                agent.merge()
            t = t + 1
            if new_borders != old_borders:
                plot_discretization(i, t)
                agent.set_limits(new_borders)
            final_state = state.number
            if config.PLOT and i % config.PLOT_EVERY == 0:
                plot_path(state)
        for j in range(config.VALUE_ITERATIONS):
            agent.update_policy()
        end = time.time()
        print("Episode " + str(i) + " finished! Time="+str(int(end-start)) + "seconds")
        f.write("Episode "+str(i)+" State #"+str(final_state)+" Total reward: "+str(total_reward)+"\n")
        f.close()
        if config.PLOT and i % config.PLOT_EVERY == 0:
            # Episode path
            path_fig.savefig(path_name)
            plt.close(path_name)
            # Discretization map
            plot_discretization(i, t)
            # Heat map
            plot_heatmap(i)
            mpl.style.use('default')
    if config.SAVE is True:
        outfile = open("PickledAgent", 'wb')
        pickle.dump(agent, outfile)
        outfile.close()
        f = open('state_visits.npy', 'wb')
        np.save(f, n_visits)
        f.close()
    game.close()
