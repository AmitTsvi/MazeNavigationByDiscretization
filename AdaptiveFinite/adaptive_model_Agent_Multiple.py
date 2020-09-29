import numpy as np
from src import agent
from tree_model_based_multiple import Node, Tree


class AdaptiveModelBasedDiscretization(agent.FiniteHorizonAgent):

    def __init__(self, epLen, flag, rmax):
        '''args:
            epLen - number of trees
        '''

        self.epLen = epLen
        self.flag = flag

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(epLen):
            tree = Tree(flag, rmax)
            self.tree_list.append(tree)

        self.rmax = rmax

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(self.epLen):
            tree = Tree(self.flag, self.rmax)
            self.tree_list.append(tree)

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, raw_action, reward, newObs, timestep, active_node):
        '''Add observation to records'''

        ''' Gets the tree that was used at that specific timestep '''
        tree = self.tree_list[timestep]

        # Increments the number of visits
        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        # Add sample to saved samples
        active_node.samples.append(obs+(raw_action,)+(reward,))

        # Update empirical estimate of average reward for that node
        if active_node.num_unique_visits == 32:
            active_node.rEst = np.average([s[4] for s in active_node.samples])
        if active_node.num_unique_visits >= 32:  # TODO: pass as argument
            active_node.rEst = ((t-1)*active_node.rEst + reward) / t
        # print('Mean reward: ' + str(active_node.rEst))

        # update transition kernel based off of new transition
        next_tree = tree
        new_obs_loc = np.argmin(np.max(np.abs(np.asarray(next_tree.state_leaves) - np.array(newObs)), axis=1))
        active_node.pEst[new_obs_loc] += 1

        # determines if it is time to split the current ball
        if t >= 4**active_node.num_splits:  # TODO: a wrong threshold, opened issue on git to clarify
            children = tree.split_node(active_node, 10, tree)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Solves the empirical Bellman equations
        for h in np.arange(self.epLen-1,-1,-1):
            # Gets the current tree for this specific time step
            tree = self.tree_list[h]
            for node in tree.tree_leaves:
                # If the node has not been visited before - set its Q Value
                # to be optimistic

                # Otherwise solve for the Q Values with the bonus term
                next_tree = tree
                vEst = np.dot((np.asarray(node.pEst)) / (np.sum(np.array(node.pEst))), next_tree.vEst)
                node.qEst = node.rEst + vEst

                if node.num_unique_visits < 32:  # TODO: pass as argument
                    node.qVal = self.rmax
                else:
                    node.qVal = node.qEst

            # After updating the Q Value for each node - computes the estimate of the value function
            index = 0
            for state_val in tree.state_leaves:  # TODO: V can get 2RMAX from artifical q. in uniform we made sure it won't happen
                _, qMax = tree.get_active_ball_for_update(state_val)
                tree.vEst[index] = qMax
                index += 1

        self.greedy = self.greedy
        pass

    def split_ball(self, node):
        children = self.node.split_ball()
        pass

    def greedy(self, state, timestep):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
            active_node - selected node for taking action
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = np.random.uniform(active_node.action_val[0] - active_node.radius, active_node.action_val[0] + active_node.radius)
        return action, active_node

    def pick_action(self, state, timestep):
        action, active_node = self.greedy(state, timestep)
        return action, active_node
