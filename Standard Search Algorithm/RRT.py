# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*
#Gokul Srinivasan

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag
    ###########################################        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)
    ###########################################
    
    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        return np.sqrt((node1.row-node2.row)**2 + (node1.col-node2.col)**2)

    ###########################################    
    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        #To Check if the obstacle is between the points, get all points
        all_points_on_path = zip(np.linspace(node1.row, node2.row, dtype=int), np.linspace(node1.col,node2.col, dtype=int))
        #Check if any of the points is an obstacle
        for p in all_points_on_path:
            if self.map_array[p[0]][p[1]] == 0:
                return True
        return False

    ###########################################
    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        if np.random.random() < goal_bias:
            pt = [self.goal.col,self.goal.row]
        else:
            pt = [np.random.randint(0,self.size_col-1), np.random.randint(0, self.size_row-1)]
        return pt

    ###########################################    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        sample =[[vertice.row,vertice.col] for vertice in self.vertices]
        kdtree = spatial.cKDTree(sample)
        _,index = kdtree.query(point)
        return self.vertices[index]

    ###########################################
    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        ### YOUR CODE HERE ###
        # Use kdtree to find the neighbors within neighbor size
        samples = [[v.row, v.col] for v in self.vertices]
        kdtree = spatial.cKDTree(samples)
        ind = kdtree.query_ball_point([new_node.row, new_node.col], neighbor_size)
        neighbors = [self.vertices[i] for i in ind]
        # Remove the new_node itself
        neighbors.remove(new_node)
        return neighbors
    ###########################################    
    def extend_node(self, extend_dist = 10 , goal_bias = 0.05):
        #Extend a new node to the current tree structure
        # extend_dis - extension distance for each step
        # goal_bias - the possibility of choosing the goal instead of a random point

        #Generate a new points
        new_point = self.get_new_point(goal_bias)
        nearest_node = self.get_nearest_node(new_point)
        #Calculae new node location
        slope = np.arctan2(new_point[1]-nearest_node.col, new_point[0]-nearest_node.row)
        new_row = nearest_node.row + extend_dist*np.cos(slope)
        new_col = nearest_node.col + extend_dist*np.sin(slope)
        new_node = Node(int(new_row), int(new_col))

        # Check boundary and collision
        if (0 <= new_row < self.size_row) and (0 <= new_col < self.size_col) and not self.check_collision(nearest_node, new_node):
            # If pass, add the new node
            new_node.parent = nearest_node
            new_node.cost = extend_dist
            self.vertices.append(new_node)

            # Check if goal is close
            if not self.found:
                d = self.dis(new_node, self.goal)
                if d < extend_dist:
                    self.goal.cost = d
                    self.goal.parent = new_node
                    self.vertices.append(self.goal)
                    self.found = True

            return new_node
        else:
            return None
    ###########################################
    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        # If no neighbors, skip
        if neighbors == []:
            return

        # Compute the distance from the new node to the neighbor nodes
        distances = [self.dis(new_node, node) for node in neighbors]

        # Rewire the new node
        # compute the least potential cost
        costs = [d + self.costofpath(self.start, neighbors[i]) for i, d in enumerate(distances)]
        indices = np.argsort(np.array(costs))
        # check collision and connect the best node to the new node
        for i in indices:
            if not self.check_collision(new_node, neighbors[i]):
                new_node.parent = neighbors[i]
                new_node.cost = distances[i]
                break

        # Rewire new_node's neighbors
        for i, node in enumerate(neighbors):
            # new cost
            new_cost = self.costofpath(self.start, new_node) + distances[i]
            # if new cost is lower
            # and there is no obstacles in between
            if self.costofpath(self.start, node) > new_cost and \
               not self.check_collision(node, new_node):
                node.parent = new_node
                node.cost = distances[i]
    ###########################################    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, awhilex = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        awhilex.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()

    ###########################################
    def costofpath(self,start_node,end_node):
        cost = 0
        curr_node = end_node
        while start_node.row != curr_node.row or start_node.col != curr_node.col:
            # Keep tracing back until finding the start_node 
            # or no path exists
            parent = curr_node.parent
            if parent is None:
                print("Invalid Path")
                return 0
            cost += curr_node.cost
            curr_node = parent
        
        return cost

    ###########################################
    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        for i in range(n_pts):
            new_node = self.extend_node(11,0.1)
            if self.found:
                break
        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.costofpath(self.start,self.goal)
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        # Draw result
        self.draw_map()

    ###########################################
    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.
        # Start searching       
        for i in range(n_pts):
            # Extend a new node
            new_node = self.extend_node(10, 0.1)
            # Rewire
            if new_node is not None:
                neighbors = self.get_neighbors(new_node, neighbor_size)
                self.rewire(new_node, neighbors)
        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.costofpath(self.start,self.goal)
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
