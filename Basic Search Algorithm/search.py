# Basic searching algorithms
# Class for each node in the grid
#Name : Gokul Srinivasan

class Node:
	def __init__(self, row, col, is_obs, h):
		self.row = row        # coordinate
		self.col = col        # coordinate
		self.is_obs = is_obs  # obstacle?
		self.g = None         # cost to come (previous g + moving cost)
		self.h = h            # heuristic
		self.cost = None      # total cost (depend on the algorithm)
		self.parent = None    # previous node

#########################################
def getNeighbours(node, grid, setParent = True):
	row, col = node.row, node.col
	neighbours = [(row, col+1), (row+1, col), (row, col-1), (row-1, col)]
	Neighbours = []
	for i in neighbours:
		if not (i[0]<0 or i[1]<0 or i[0]>=len(grid) or i[1]>=len(grid[0]) or grid[i[0]][i[1]]==1):
			Neighbours.append(Node(i[0],i[1],0,0))
  
	for k in Neighbours:
		if setParent:
			k.parent = node        
	return Neighbours
########################################   
def checkifnodeExists(node, node_list):
	for i in node_list:
		if (i.row, i.col) == (node.row, node.col):
			return True
	return False
###########################################
def travel(path,curr):
	x = curr	
	while(x.parent):		
		path = travel(path, x.parent)
		path.append([x.parent.row,x.parent.col])	
		break	
	return path 
###########################################	
def bfs(grid, start, goal):
	'''Return a path found by BFS alogirhm 
	   and the number of steps it takes to find it.

	arguments:
	grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
		   e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
	start - The start node in the map. e.g. [0, 0]
	goal -  The goal node in the map. e.g. [2, 2]

	return:
	path -  A nested list that represents coordinates of each step (including start and goal node), 
			with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
	steps - Number of steps it takes to find the final solution, 
			i.e. the number of nodes visited before finding a path (including start and goal node)

	# >>> from main import load_map
	# >>> grid, start, goal = load_map('test_map.csv')
	# >>> bfs_path, bfs_steps = bfs(grid, start, goal)
	It takes 10 steps to find a path using BFS
	# >>> bfs_path
	[[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
	'''
	### YOUR CODE HERE ###
	path = []
	steps = 0
	found = False
	#List of all visited nodes and priority queue
	visited, queue = [],[]
	#Define start node
	Start = Node(start[0], start[1], grid[start[0]][start[1]],0)
	visited.append(Start)
	steps += 1
	#Add the start node to the queue
	queue.append(Start)
	#Start of the search loop
	while queue:
		if found:
			break
		currentNode = queue.pop(0)
		#Get all the valid neighbour node
		for neighbour in getNeighbours(currentNode, grid):
			if not checkifnodeExists(neighbour, visited):
				steps += 1
				visited.append(neighbour)
				#Check if the node is goal or not.
				if ([neighbour.row, neighbour.col] == goal):
					found = True
					#BackTracking  					               
					path = travel(path,neighbour)
					path.append(goal)
					break
				queue.append(neighbour)

	if found:
		print(f"It takes {steps} steps to find a path using BFS")
	else:
		print("No path found")
	return path, steps
#########################################
def dfs(grid, start, goal):
	'''Return a path found by DFS alogirhm 
	   and the number of steps it takes to find it.

	arguments:
	grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
		   e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
	start - The start node in the map. e.g. [0, 0]
	goal -  The goal node in the map. e.g. [2, 2]

	return:
	path -  A nested list that represents coordinates of each step (including start and goal node), 
			with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
	steps - Number of steps it takes to find the final solution, 
			i.e. the number of nodes visited before finding a path (including start and goal node)

	>>> from main import load_map
	>>> grid, start, goal = load_map('test_map.csv')
	>>> dfs_path, dfs_steps = dfs(grid, start, goal)
	It takes 9 steps to find a path using DFS
	>>> dfs_path
	[[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
	'''
	### YOUR CODE HERE ###
	def DFS_recursive(visitedNodes, grid, node, step, goal, found):
		path = []
		if not checkifnodeExists(node, visitedNodes):
			visitedNodes.append(node)
			step += 1
			for neighbour in getNeighbours(node, grid):
				if ([neighbour.row, neighbour.col] == goal):
					found = True
					#The goal must also  be counted as step.
					# If the neighbour is not the goal, it will get updated when called again.
					step += 1    
					
					path = travel(path,neighbour)
					path.append(goal)
					return [found, path, step]
				else:
					[found, path, step] = DFS_recursive(visitedNodes, grid, neighbour, step, goal, found)
				if found:
					return [found, path, step]
		return [found, path, step]
	
	path = []
	steps = 0
	found = False

	visited = [] 
	Start = Node(start[0], start[1], grid[start[0]][start[1]],0)
	[found, path, steps] = DFS_recursive(visited, grid, Start, steps, goal, found)

	if found:
		print(f"It takes {steps} steps to find a path using DFS")
	else:
		print("No path found")
	return path, steps
###########################################
def Distance(start, goal):
	# Calculates and returns the Manhattan Distance
	# |𝑛𝑜𝑑𝑒 𝑥 − 𝑔𝑜𝑎𝑙 𝑥 | + |𝑛𝑜𝑑𝑒 𝑦 − 𝑔𝑜𝑎𝑙 𝑦 |
	return (abs(goal[0] - start[0]) + abs(goal[1] - start[1]))
###########################################
def dijkstra(grid, start, goal):
	'''Return a path found by Dijkstra alogirhm 
	   and the number of steps it takes to find it.

	arguments:
	grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
		   e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
	start - The start node in the map. e.g. [0, 0]
	goal -  The goal node in the map. e.g. [2, 2]

	return:
	path -  A nested list that represents coordinates of each step (including start and goal node), 
			with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
	steps - Number of steps it takes to find the final solution, 
			i.e. the number of nodes visited before finding a path (including start and goal node)

	>>> from main import load_map
	>>> grid, start, goal = load_map('test_map.csv')
	>>> dij_path, dij_steps = dijkstra(grid, start, goal)
	It takes 10 steps to find a path using Dijkstra
	>>> dij_path
	[[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
	'''
	### YOUR CODE HERE ###
	path = []
	steps = 0
	found = False
	visited, priorityqueue = [], []
	
	Start = Node(start[0], start[1], grid[start[0]][start[1]],0)
	Start.cost, tC, steps = 0,1,-1
	priorityqueue.append(Start)
	
	while priorityqueue:
		if found:
			break
		priorityqueue = sorted(priorityqueue, key = lambda node: node.cost)
		curr = priorityqueue.pop(0)
		steps += 1
		row , col = curr.row, curr.col
		for a,b in ((row, col+1), (row+1, col), (row, col-1), (row-1, col)):
			if [row, col] == goal:
				found = True					
				path = travel(path,curr)
				path.append(goal)				
				break
			elif 0 <= a< len(grid) and 0 <= b < len(grid[0]) and found == False:
				newNode = Node(a,b,grid[a][b], h=0)
				if grid[a][b] == 0:
					newNode.parent = curr
					newNode.cost = newNode.parent.cost + tC
					notpresent = True
					for existingNode in priorityqueue:
						if([existingNode.row, existingNode.col] == [newNode.row,newNode.col]):
							if existingNode.cost > newNode.cost: 
								notpresent = False
								existingNode.cost = newNode.cost
								break
					if notpresent is True and (a,b) not in visited:
						priorityqueue.append(newNode)
						visited.append((newNode.row, newNode.col))
		if found:
			break

	if found:
		print(f"It takes {steps} steps to find a path using Dijkstra")
	else:
		print("No path found")
	return path, steps
###########################################
def astar(grid, start, goal):
	'''Return a path found by A* alogirhm 
	   and the number of steps it takes to find it.

	arguments:
	grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
		   e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
	start - The start node in the map. e.g. [0, 0]
	goal -  The goal node in the map. e.g. [2, 2]

	return:
	path -  A nested list that represents coordinates of each step (including start and goal node), 
			with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
	steps - Number of steps it takes to find the final solution, 
			i.e. the number of nodes visited before finding a path (including start and goal node)

	>>> from main import load_map
	>>> grid, start, goal = load_map('test_map.csv')
	>>> astar_path, astar_steps = astar(grid, start, goal)
	It takes 7 steps to find a path using A*
	>>> astar_path
	[[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
	'''
	### YOUR CODE HERE ###
	path = []
	steps = 0
	found = False
	visited = []
	priorityqueue = []
	Start = Node(start[0], start[1], grid[start[0]][start[1]],h = 0)
	Start.g , Start.cost = Distance(start,start),0
	h = Distance(start,goal)
	priorityqueue.append(Start)
	
	while priorityqueue:
		if found:
			break
		priorityqueue = sorted(priorityqueue, key = lambda node: node.cost)
		curr = priorityqueue.pop(0)
		steps += 1
		row , col = curr.row, curr.col
		for a,b in ((row, col+1), (row+1, col), (row, col-1), (row-1, col)):
			if [row, col] == goal:
				found = True					
				path = travel(path,curr)
				path.append(goal)				
				break
			elif 0 <= a< len(grid) and 0 <= b < len(grid[0]) and found == False:
				h = Distance([a,b],goal)
				newNode = Node(a,b,grid[a][b], h)
				if grid[a][b] == 0:
					newNode.parent = curr
					newNode.g = curr.g + Distance([a,b],[row,col])
					newNode.cost = newNode.g + newNode.h
					notpresent = True
					for existingNode in priorityqueue:
						if([existingNode.row, existingNode.col] == [newNode.row,newNode.col]):
							if existingNode.cost > newNode.cost: 
								notpresent = False
								existingNode.cost = newNode.cost
								break
					if notpresent is True and (a,b) not in visited:
						priorityqueue.append(newNode)
						visited.append((newNode.row, newNode.col))
		if found:
			break

	if found:
		print(f"It takes {steps} steps to find a path using A*")
	else:
		print("No path found")
	return path, steps
###########################################
# Doctest
if __name__ == "__main__":
	# load doc test
	from doctest import testmod, run_docstring_examples
	# Test all the functions
	testmod()
