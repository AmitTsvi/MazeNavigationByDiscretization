import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rnd

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''


class Node():
    def __init__(self, qVal, rEst, pEst, num_visits, num_unique_visits, num_splits, state_val, action_val, radius, rmax):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        self.qVal = qVal
        self.rEst = rEst
        self.pEst = pEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = action_val
        self.radius = radius
        self.children = None
        self.rmax = rmax
        self.samples = []
        self.qEst = 0

    def is_sample_in_child(self, s):
        if np.max(np.abs(np.asarray(s[0:3]) - np.asarray(self.state_val))) <= self.radius:
            if np.max(np.abs(np.asarray(s[3]) - np.asarray(self.action_val))) <= self.radius:
                return True
        return False

        # Splits a node by covering it with 16 children, as here S times A is [0,1]^4
        # each with half the radius
    def split_node(self, flag):
        rh = self.radius/2
        # creation of the children
        self.children = [Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, 0, self.num_splits+1,
                              (self.state_val[0]+k0*rh, self.state_val[1]+k1*rh, self.state_val[2]+k2*rh),
                              (self.action_val[0]+k3*rh,), rh, self.rmax) for k0 in [-1,1] for k1 in [-1,1] for k2 in [-1,1] for k3 in [-1, 1]]
        # calculating better r and q estimates based on the sample partition
        for child in self.children:
            child.samples = [s for s in self.samples if child.is_sample_in_child(s)]
            child.num_unique_visits = len(child.samples)
            if child.num_unique_visits > 0:
                child.rEst = np.average([s[4] for s in child.samples])
            else:
                child.rEst = self.rmax
                child.pEst = np.zeros(len(self.pEst)).tolist()
            if child.num_unique_visits >= 32:  # TODO: pass as argument
                child.qVal = self.qVal  # TODO: think
            else:
                child.qVal = 2*self.rmax

        # clearing fathers samples
        self.samples.clear()
        return self.children


class Tree():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, flag, rmax):
        self.head = Node(2*rmax, rmax, [0], 0, 0, 0, (0.5, 0.5, 0.5), (0.5,), 0.5, rmax)
        self.flag = flag
        self.state_leaves = [(0.5, 0.5, 0.5)]
        self.vEst = [0]
        self.tree_leaves = [self.head]
        self.rmax = rmax

    # Returns the head of the tree
    def get_head(self):
        return self.head

    def split_node(self, node, timestep, previous_tree):
        children = node.split_node(self.flag)

        # Update the list of leaves in the tree
        self.tree_leaves.remove(node)
        for child in children:
            self.tree_leaves.append(child)

        # Gets one of their state value
        child_1_state = children[0].state_val
        child_1_radius = children[0].radius

        # Determines if we also need to adjust the state_leaves and carry those estimates down as well
        if np.min(np.max(np.abs(np.asarray(self.state_leaves) - np.array(child_1_state)), axis=1)) >= child_1_radius:
            # find parents place in state_leaves and in vEst
            parent = node.state_val
            parent_index = self.state_leaves.index(parent)
            parent_vEst = self.vEst[parent_index]

            # remove parent from leaves vectors
            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            # appending unique state_values of the new children
            unique_state_values = list(set([child.state_val for child in children]))
            for unique_state_val in unique_state_values:
                self.state_leaves.append(unique_state_val)
                # childvEst = np.max([child.qVal for child in children if child.state_val == unique_state_val])
                self.vEst.append(0)  # TODO: think

            # Lastly we need to adjust the transition kernel estimates from the previous tree
            if timestep >= 1:
                previous_tree.update_transitions_after_split(parent_index, 8)

        return children

    def update_transitions_after_split(self, parent_index, num_children):
        for node in self.tree_leaves:
            # removing parent transition prob
            pEst_parent = node.pEst[parent_index]
            node.pEst.pop(parent_index)
            # adding and normalizing transition prob for each unique state_val child
            for i in range(num_children):
                node.pEst.append(pEst_parent/num_children)

    # Plot function which plots the tree on a graph on [0,1]^2 with the discretization
    def plot(self, fig):
        ax = plt.gca()
        self.plot_node(self.head, ax)
        plt.xlabel('X Space')
        plt.ylabel('Y Space')
        return fig

    # Recursive method which plots all subchildren
    def plot_node(self, node, ax):
        if node.children == None:
            # print('Child Node!')
            rect = patches.Rectangle((node.state_val[0] - node.radius, node.state_val[1] - node.radius),
                                     node.radius * 2, node.radius * 2, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # plt.text(node.state_val, node.action_val, np.around(node.qVal, 3))
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            _, qEst = self.get_active_ball_for_update(node.state_val)
            text = str(round(qEst, 1))
            ax.annotate(text, (cx, cy), color='b', weight='light', fontsize=5, ha='center', va='center')
        else:
            for child in node.children:
                self.plot_node(child, ax)

    # Recursive method which gets number of subchildren
    def get_num_balls(self, node):
        num_balls = 0
        if node.children == None:
            return 1
        else:
            for child in node.children:
                num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        return self.get_num_balls(self.head)

    # A method which implements recursion and greedily selects the selected ball
    # to have the largest qValue and contain the state being considered

    def get_active_ball_recursion(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node, node.qVal
        else:
            # Otherwise checks each child node
            qVal = -np.inf  # TODO: think about this value (was 0)
            for child in node.children:
                # if the child node contains the current state
                if self.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qVal = self.get_active_ball_recursion(state, child)
                    if new_qVal > qVal:
                        active_node, qVal = new_node, new_qVal
                    elif new_qVal == qVal:
                        r = rnd.randrange(2)
                        if r == 0:
                            active_node, qVal = new_node, new_qVal
                else:
                    pass
        return active_node, qVal

    def get_active_ball(self, state):
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    def get_active_ball_recursion_for_update(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node, node.qEst
        else:
            # Otherwise checks each child node
            qEst = -np.inf  # TODO: think about this value (was 0)
            for child in node.children:
                # if the child node contains the current state
                if self.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qEst = self.get_active_ball_recursion_for_update(state, child)
                    if new_qEst > qEst:
                        active_node, qEst = new_node, new_qEst
                    elif new_qEst == qEst:
                        r = rnd.randrange(2)
                        if r == 0:
                            active_node, qEst = new_node, new_qEst
                else:
                    pass
        return active_node, qEst

    def get_active_ball_for_update(self, state):
        active_node, qEst = self.get_active_ball_recursion_for_update(state, self.head)
        return active_node, qEst

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return np.max(np.abs(np.asarray(state) - np.asarray(node.state_val))) <= node.radius
