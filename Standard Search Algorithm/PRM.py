# Standard Algorithm Implementation
# Sampling-based Algorithms PRM
#Gokul Srinivasan

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial

samp_meth = 1
# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

    ###########################################
    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        #To Check if the obstacle is between the points, get all points
        all_points_on_path = zip(np.linspace(p1[0], p2[0], dtype=int), np.linspace(p1[1], p2[1], dtype=int))
        #Check if any of the points is an obstacle
        for p in all_points_on_path:
            if self.map_array[p[0]][p[1]] == 0:
                return True
        return False

    ###########################################
    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        #euclidean distance = sqrt(sum of all (yi-xi)^2)
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    ###########################################
    def get_sample_points(self, n_pts, random=True):
        '''Get the row and col coordinates of n sample points
        arguments:
            n_pts - total number of points to be sampled
            random - random or uniform?

        return:
            p_row - row coordinates of n sample points (1D)
            p_col - col coordinates of n sample points (1D)
        '''
        # number of points
        n_row = int(np.sqrt(n_pts * self.size_row / self.size_col))
        n_col = int(n_pts / n_row)
        # generate uniform points
        if not random:
            sample_row = np.linspace(0, self.size_row-1, n_row, dtype=int)
            sample_col = np.linspace(0, self.size_col-1, n_col, dtype=int)
            p_row, p_col = np.meshgrid(sample_row, sample_col)
            p_row = p_row.flatten()
            p_col = p_col.flatten()
        # generate random points
        else:
            p_row = np.random.randint(0, self.size_row-1, n_pts, dtype=int)
            p_col = np.random.randint(0, self.size_col-1, n_pts, dtype=int)
        return p_row, p_col
    # def get_random_sample_pts(self,n_pts):
    #     #Returns the row and colun coordinates for random sample points
    #     # number of points
    #     # row = int(np.sqrt(n_pts * self.size_row / self.size_col))
    #     # col = int(n_pts / row)
    #     p_row = np.random.randint(0, self.size_row-1, n_pts, dtype=int)
    #     p_col = np.random.randint(0, self.size_col-1, n_pts, dtype=int)
    #     return p_row, p_col

    # def get_uniform_sample_pts(self, n_pts):
    #     #Returns the row and colun coordinates for uniformly sampled points
    #     row = int(np.sqrt(n_pts * self.size_row / self.size_col)) #number of ros
    #     col = int(n_pts / row) #number of columns
    #     row_sample = np.linspace(0, self.size_row-1, row, dtype=int)
    #     col_sample = np.linspace(0, self.size_col-1, col, dtype=int)
    #     p_row, p_col = np.meshgrid(row_sample, col_sample)
    #     p_row = p_row.flatten()
    #     p_col = p_col.flatten()    
    #     return p_row, p_col

    ###########################################
    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        ### YOUR CODE HERE ###

        row,col = self.get_sample_points(n_pts,random = False) #Generate uniform points
        #Chcking for obstacles
        for r,c in zip (row,col):
            if self.map_array[r][c] == 1:
                self.samples.append((r, c))

    ###########################################
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        row,col = self.get_sample_points(n_pts)
        # row,col = self.get_random_sample_pts(n_pts) #Generate random points
        #Chcking for obstacles
        for r,c in zip (row,col):
            if self.map_array[r][c] == 1:
                self.samples.append((r, c))

    ###########################################
    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        row_1,col_1 = self.get_sample_points(n_pts)
        # row_1,col_1 = self.get_random_sample_pts(n_pts) #Generate random points
        # Generate random points at some distance from the preivous generated points
        scale = 10 # std
        row_2 = row_1 + np.random.normal(0.0, scale, n_pts).astype(int)
        col_2 = col_1 + np.random.normal(0.0, scale, n_pts).astype(int)
        #Checking if the point is close to an obstacle
        for r1,c1,r2,c2 in zip (row_1,col_1,row_2,col_2):
            if not(0 <= r2 < self.size_row) or not(0 <= c2 < self.size_col):
                continue
            if self.map_array[r1][c1] == 1 and self.map_array[r2][c2] == 0:
                self.samples.append((r1, c1))
            elif self.map_array[r1][c1] == 0 and self.map_array[r2][c2] == 1:
                self.samples.append((r2, c2))    
        
    ###########################################
    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        row_1,col_1 = self.get_sample_points(n_pts)
        # row_1,col_1 = self.get_random_sample_pts(n_pts) #Generate random points
        # Generate random points at some distance from the preivous generated points
        scale = 20 # std
        row_2 = row_1 + np.random.normal(0.0, scale, n_pts).astype(int)
        col_2 = col_1 + np.random.normal(0.0, scale, n_pts).astype(int)
        #Checking if the point is close to an obstacle
        for r1,c1,r2,c2 in zip (row_1,col_1,row_2,col_2):
            if ((not(0 <= c2 < self.size_col) or not(0 <= r2 < self.size_row)) or self.map_array[r2][c2] == 0) and self.map_array[r1][c1] == 0:
                # make sure the midpoint is inside the map
                mid_row, mid_col = int(0.5*(r1+r2)), int(0.5*(c1+c2))
                if 0 <= mid_row < self.size_row and 0 <= mid_col < self.size_col and self.map_array[mid_row][mid_col] == 1:
                    self.samples.append((mid_row, mid_col))

    ###########################################
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    ###########################################
    def add_vertices_pair_edges(self,pairs):
    # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        # self.graph.add_nodes_from([])
        # self.graph.add_weighted_edges_from(pairs)        
        for pair in pairs:
            if pair[0] == "start":
                p1 = self.samples[-2]
            elif pair[0] == "goal":
                p1 = self.samples[-1]
            else:
                p1 = self.samples[pair[0]]
            p2 = self.samples[pair[1]]
            if not self.check_collision(p1,p2):
                d = self.dis(p1,p2)
                edge = [(pair[0],pair[1],d)]
                self.graph.add_weighted_edges_from(edge)
    
    ###########################################
    def connect_vertices(self, kdtree_d=20):
        '''Add nodes and edges to the graph from sampled points
        arguments:
            kdtree_d - the distance for kdtree to search for nearest neighbors

        Add nodes to graph from self.samples
        Build kdtree to find neighbor pairs and add them to graph as edges
        '''
        # Finds k nearest neighbors
        # kdtree 
        self.kdtree = spatial.cKDTree(list(self.samples))
        pairs = self.kdtree.query_pairs(kdtree_d)

        # Add the neighbor to graph
        self.graph.add_nodes_from(range(len(self.samples)))
        self.add_vertices_pair_edges(pairs)


    ###########################################
    # def sample(self, n_pts=1000, sampling_method="uniform"):
    def sample(self,n_pts, sampling_method,kdtree_d=20):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []
        global samp_meth
        # Sample methods
        if sampling_method == "uniform":
            samp_meth = 1
            kdtree_d = 15
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            samp_meth = 2
            kdtree_d = 18
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            samp_meth = 3
            kdtree_d = 20
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            samp_meth = 4
            kdtree_d = 20
            self.bridge_sample(n_pts)

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        # pairs = []
        #Finding K nearest neighbors with kdtree
        # kdtree_d = 20
        # self.kdtree = spatial.cKDTree(list(self.samples))
        # pairs = self.kdtree.query_pairs(kdtree_d)
        # #Add neighbor to graph
        # self.graph.add_nodes_from(range(len(self.samples)))
        # self.add_vertices_pair_edges(pairs)
         
        self.connect_vertices(kdtree_d)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))

    ###########################################
    def search(self, start, goal,kdtree_d = 15):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []
        # kdtree_d = 20
        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###
        

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        global samp_meth
        if samp_meth == 1 or samp_meth == 2:
            kdtree_d = 18
        elif samp_meth == 3 or samp_meth == 4:
            kdtree_d = 200
            
        
        start_goal_tree = spatial.cKDTree([start,goal])
        neighbors = start_goal_tree.query_ball_tree(self.kdtree,kdtree_d)
        start_pairs = ([['start', neighbor] for neighbor in neighbors[0]])
        goal_pairs = ([['goal', neighbor] for neighbor in neighbors[1]])

        # Add the edge to graph
        self.add_vertices_pair_edges(start_pairs)
        self.add_vertices_pair_edges(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
            
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        