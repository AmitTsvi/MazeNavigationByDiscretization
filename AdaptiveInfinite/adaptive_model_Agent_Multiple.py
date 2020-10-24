import numpy as np
from tree_model_based_multiple import Tree
import config


class AdaptiveModelBasedDiscretization:
    def __init__(self, num_actions):
        """args: epLen - number of trees"""
        self.tree = Tree(num_actions)
        self.num_actions = num_actions
        self.limits = (0, 0, 0, 0)

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree = Tree(self.num_actions)

    def update_obs(self, obs, raw_action, reward, newObs, timestep, active_node):
        """Add observation to records"""
        # Increments the number of visits
        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        # Add sample to saved samples
        active_node.samples.append(obs+(raw_action,)+(reward,)+newObs)

        # Update empirical estimate of average reward for that node
        if active_node.num_unique_visits == config.VISITS_TO_KNOWN:
            active_node.rEst = np.average([s[4] for s in active_node.samples])
        if active_node.num_unique_visits >= config.VISITS_TO_KNOWN:
            active_node.rEst = ((t-1)*active_node.rEst + reward) / t
        # print('Mean reward: ' + str(active_node.rEst))

        # update transition kernel based off of new transition
        basic_dist_array = np.abs(np.asarray(self.tree.state_leaves) - np.array(newObs))
        min_x_y_dist = np.min(np.sum((basic_dist_array[:, 0:2])**2, axis=1))
        min_x_y_indices = np.where(np.sum((basic_dist_array[:, 0:2])**2, axis=1) == min_x_y_dist)[0]
        possible_entries = [basic_dist_array[i][2] for i in min_x_y_indices]
        min_theta_index = np.argmin(possible_entries)
        new_obs_loc = min_x_y_indices[min_theta_index]
        active_node.pEst[new_obs_loc] += 1

        # determines if it is time to split the current ball
        if t >= 4**active_node.num_splits and active_node.num_splits < config.MAX_SPLITS:  # TODO: can change threshold
            children = self.tree.split_node(active_node)

    def update_policy(self):
        """Update internal policy based upon records"""
        # Solves the empirical Bellman equations
        for node in self.tree.tree_leaves:
            # If the node has not been visited before - set its Q Value
            # to be optimistic

            # Otherwise solve for the Q Values with the bonus term
            psum = np.sum(np.array(node.pEst))
            if psum > 0 and node.num_unique_visits >= config.VISITS_TO_KNOWN:
                vEst = np.dot((np.asarray(node.pEst)) / (psum), self.tree.vEst)
            else:
                vEst = 0
            node.td_error = abs(node.qEst - (node.rEst + vEst))
            node.qEst = node.rEst + vEst

            if node.num_unique_visits < config.VISITS_TO_KNOWN:
                node.qVal = 2*config.R_MAX
            else:
                node.qVal = node.qEst

        # After updating the Q Value for each node - computes the estimate of the value function
        index = 0
        for state_val in self.tree.state_leaves:
            _, qMax = self.tree.get_active_ball_for_update(state_val)
            self.tree.vEst[index] = qMax
            index += 1

        self.greedy = self.greedy
        pass

    def greedy(self, state):
        """
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
            active_node - selected node for taking action
        """
        # Gets the selected ball
        active_node, qVal = self.tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = active_node.action_val[0]
        return action, active_node

    def pick_action(self, state):
        action, active_node = self.greedy(state)
        return action, active_node

    def rescale(self, quadrant):
        tree = self.tree.rescale(quadrant)
        self.tree = tree

    def set_limits(self, limits):
        self.limits = limits

    def get_limits(self):
        return self.limits

    def merge(self):
        self.tree.merge_nodes()
