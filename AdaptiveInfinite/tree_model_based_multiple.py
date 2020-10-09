import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rnd

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''


class Node():
    def __init__(self, qVal, rEst, pEst, num_visits, num_unique_visits, num_splits, state_val, action_val, radius, rmax, num_actions, theta_radius):
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
        self.num_actions = num_actions
        self.theta_radius = theta_radius

    def is_sample_in_child(self, s):
        if np.max(np.abs(np.asarray(s[0:2]) - np.asarray(self.state_val[0:2]))) <= self.radius:
            if np.max(np.abs(np.asarray(s[2]) - np.asarray(self.state_val[2]))) <= self.theta_radius:
                if np.max(np.abs(np.asarray(s[3]) - np.asarray(self.action_val))) <= (1/(2*self.num_actions)):
                    return True
        return False

        # Splits a node by covering it with 16 children, as here S times A is [0,1]^4
        # each with half the radius
    def split_node(self, flag):
        rh = self.radius/2
        rh_theta = self.theta_radius/2
        # creation of the children
        self.children = [Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, 0, self.num_splits+1,
                              (self.state_val[0]+k0*rh, self.state_val[1]+k1*rh, self.state_val[2]+k2*rh_theta),
                              (self.action_val[0],), rh, self.rmax, self.num_actions, rh_theta) for k0 in [-1,1] for k1 in [-1,1] for k2 in [-1,1]]
        # calculating better r and q estimates based on the sample partition
        for child in self.children:
            child.samples = [s for s in self.samples if child.is_sample_in_child(s)]
            child.num_unique_visits = len(child.samples)
            if child.num_unique_visits >= 5:
                child.rEst = np.average([s[4] for s in child.samples])
            else:
                child.rEst = self.rmax
                child.pEst = np.zeros(len(self.pEst)).tolist()
            if child.num_unique_visits >= 5:  # TODO: pass as argument
                child.qVal = self.qVal  # TODO: think
            else:
                child.qVal = 2*self.rmax

        # clearing fathers samples
        self.samples.clear()
        return self.children


class Tree():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, flag, rmax, num_actions):
        self.head = Node(2*rmax, rmax, [0], 0, 0, 0, (0.5, 0.5, 0.5), (0.5,), 0.5, rmax, num_actions, 0.5)
        self.flag = flag
        self.state_leaves = [(0.5, 0.5, 0.5)]
        self.vEst = [0]
        self.tree_leaves = []
        self.rmax = rmax
        self.num_actions = num_actions
        self.head.children = self.get_initial_children()
        self.tree_leaves = self.head.children.copy()

    def get_initial_children(self):
        children = []
        centers = []
        gap = 1 / self.num_actions
        min_bord = 0.0
        for action in range(self.num_actions):
            max_bord = min_bord + gap
            center = (min_bord + max_bord)/2
            centers.append(center)
            min_bord = max_bord
        for action in range(self.num_actions):
            child = Node(2*self.rmax, self.rmax, [0], 0, 0, 0, (0.5, 0.5, 0.5), (centers[action],), 0.5, self.rmax, self.num_actions, 0.5)
            children.append(child)
        return children

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
        child_1_theta_radius = children[0].theta_radius

        # Determines if we also need to adjust the state_leaves and carry those estimates down as well
        if np.min(np.max(np.abs(np.asarray(self.state_leaves) - np.array(child_1_state)) - [child_1_radius, child_1_radius, child_1_theta_radius], axis=1)) >= 0:
            # find parents place in state_leaves and in vEst
            parent = node.state_val
            parent_index = self.state_leaves.index(parent)
            parent_vEst = self.vEst[parent_index]
            parent_unique_visits = node.num_unique_visits

            # remove parent from leaves vectors
            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            # appending unique state_values of the new children
            unique_state_values = list(set([child.state_val for child in children]))
            for unique_state_val in unique_state_values:
                self.state_leaves.append(unique_state_val)
                # childvEst = np.max([child.qVal for child in children if child.state_val == unique_state_val])
                self.vEst.append(0)  # TODO: think

            # prepare transition distribution
            dist = []
            for unique_state_val in unique_state_values:
                sum = 0
                for child in children:
                    if self.state_within_node(unique_state_val, child):
                        sum += child.num_unique_visits
                dist.append(sum/parent_unique_visits)

            # Lastly we need to adjust the transition kernel estimates from the previous tree
            if timestep >= 1:
                previous_tree.update_transitions_after_split(parent_index, 8, dist)

        return children

    def update_transitions_after_split(self, parent_index, num_children, dist):
        for node in self.tree_leaves:
            # removing parent transition prob
            pEst_parent = node.pEst[parent_index]
            node.pEst.pop(parent_index)
            # adding and normalizing transition prob for each unique state_val child
            for i in range(num_children):
                node.pEst.append(pEst_parent*dist[i])

    # Plot function which plots the tree on a graph on [0,1]^2 with the discretization
    def plot(self, fig):
        ax = plt.gca()
        self.plot_node(self.head, ax)
        plt.xlabel('X Space')
        plt.ylabel('Y Space')
        return fig

    # Recursive method which plots all subchildren
    def plot_node(self, node, ax):
        nodes_to_plot = {}
        nodes = [((node.state_val[0], node.state_val[1], node.radius), node.qEst) for node in self.tree_leaves]
        for node in nodes:
            key = node[0]
            value = node[1]
            if key in nodes_to_plot:
                nodes_to_plot[key] = max(nodes_to_plot[key], value)
            else:
                nodes_to_plot[key] = value
        for i, (k, v) in enumerate(nodes_to_plot.items()):
            rect = patches.Rectangle((k[0] - k[2], k[1] - k[2]),
                                     k[2] * 2, k[2] * 2, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            text = str(round(v, 1))
            ax.annotate(text, (cx, cy), color='b', weight='light', fontsize='xx-small', ha='center', va='center')

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
        return (np.max(np.abs(np.asarray(state[0:2]) - np.asarray(node.state_val[0:2]))) <= node.radius) and \
               (np.max(np.abs(np.asarray(state[2]) - np.asarray(node.state_val[2]))) <= node.theta_radius)

    def rescale_recursion(self, node, quadrant):
        node.pEst += [0, 0, 0]
        node.radius = node.radius / 2
        node.num_splits += 1
        x = node.state_val[0]
        y = node.state_val[1]
        theta = node.state_val[2]
        if quadrant == 1:
            node.state_val = (x/2+0.5, y/2+0.5, theta)
            if node.children == None:
                node.samples = [(s[0]/2+0.5, s[1]/2+0.5, s[2], s[3], s[4]) for s in node.samples]
        if quadrant == 2:
            node.state_val = (x/2, y/2+0.5, theta)
            if node.children == None:
                node.samples = [(s[0]/2, s[1]/2+0.5, s[2], s[3], s[4]) for s in node.samples]
        if quadrant == 3:
            node.state_val = (x/2, y/2, theta)
            if node.children == None:
                node.samples = [(s[0]/2, s[1]/2, s[2], s[3], s[4]) for s in node.samples]
        if quadrant == 4:
            node.state_val = (x/2+0.5, y/2, theta)
            if node.children == None:
                node.samples = [(s[0]/2+0.5, s[1]/2, s[2], s[3], s[4]) for s in node.samples]
        if node.children != None:
            for child in node.children:
                self.rescale_recursion(child, quadrant)

    def rescale(self, quadrant):  # TODO: maybe we need copy or deep copy if the old tree is erased automatically
        self.rescale_recursion(self.head, quadrant)
        new_tree = Tree(self.flag, self.rmax, self.num_actions)  # The new tree
        new_tree.vEst = self.vEst + [0, 0, 0]  # vEst fix
        for action in range(self.num_actions):  # first adding the old tree's children to the new one
            new_tree.head.children[action].children = [self.head.children[action]]  # TODO: if this is an empty action node then insert the state_val to state_leaves
        new_state_leaves = []  # place holder
        if quadrant == 1:
            self.state_leaves = [(s[0]/2+0.5, s[1]/2+0.5, s[2]) for s in self.state_leaves]  # rescaling the old ones
            new_state_leaves = [(0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.25, 0.75, 0.5)]
        if quadrant == 2:
            self.state_leaves = [(s[0]/2, s[1]/2+0.5, s[2]) for s in self.state_leaves]
            new_state_leaves = [(0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.75, 0.75, 0.5)]
        if quadrant == 3:
            self.state_leaves = [(s[0]/2, s[1]/2, s[2]) for s in self.state_leaves]
            new_state_leaves = [(0.75, 0.25, 0.5), (0.25, 0.75, 0.5), (0.75, 0.75, 0.5)]
        if quadrant == 4:
            self.state_leaves = [(s[0]/2+0.5, s[1]/2, s[2]) for s in self.state_leaves]
            new_state_leaves = [(0.25, 0.25, 0.5), (0.25, 0.75, 0.5), (0.75, 0.75, 0.5)]
        new_tree.state_leaves = self.state_leaves + new_state_leaves  # combining to the new tree
        new_nodes = []
        for action in range(self.num_actions):
            action_val = new_tree.head.children[action].action_val
            action_nodes = [Node(2*self.rmax, self.rmax, np.zeros(len(new_tree.vEst)).tolist(), 0, 0, 1,
                                new_state_leaf, action_val, 0.25, self.rmax, self.num_actions, 0.5)
                                for new_state_leaf in new_state_leaves]  # accumulating new nodes
            new_tree.head.children[action].children += action_nodes  # adding for each action branch the right nodes
            new_nodes += action_nodes
        new_tree.tree_leaves = self.tree_leaves + new_nodes  # combining to the new tree
        return new_tree
