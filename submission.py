# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        (priority, count, data) = heapq.heappop(self.queue)
        return (priority, data)

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        #raise NotImplementedError
        self.counter += 1
        priority = node[0]
        data = node[1]
        tmpNode = (priority, self.counter, data)
        heapq.heappush(self.queue, tmpNode)

        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start == goal:
        return []
    frontier = [(start, [])] #initialize frontier
    explored = set()
    while frontier:
        curr = frontier.pop(0)
        explored.add(curr[0]) #add curr state to explored
        for i in sorted(graph.neighbors(curr[0]), key=lambda x:x[0]):
            if i == goal:
                return [start] + curr[1] + [i]
            if i not in explored:
                frontier.append((i, curr[1] + [i]))
                explored.add(i)

    return None


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    #node = (cost, node)
    if start == goal:
        return []
    frontier = PriorityQueue() #initialize frontier
    frontier.append((0, start))
    explored = set()
    path = {start: [start]} #dict to keep track of path
    cost = {start: 0} #dict to keep track of cost
    while frontier:
        curr = frontier.pop()
        currCost = curr[0] 
        curr = curr[1]
        if curr not in explored:
            explored.add(curr) #add curr state to explored
            
            if curr == goal: #terminating condition
                return path[curr]
            for i in sorted(graph.neighbors(curr), key=lambda x:graph.get_edge_weight(curr, x)):
                newCost = currCost + graph.get_edge_weight(curr, i) 
                if i not in cost or newCost < cost[i]:
                    cost[i] = newCost
                    frontier.append((newCost, i))
                    path[i] = path[curr] + [i]
    return None


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    #raise NotImplementedError
    vNode = graph.nodes[v]['pos']
    goalNode = graph.nodes[goal]['pos']
    x1, y1 = vNode
    x2, y2 = goalNode
    d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return d


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError

    if start == goal:
        return []
    frontier = PriorityQueue() #initialize frontier
    frontier.append((0, start))
    explored = set()
    path = {start: [start]}
    cost = {start: 0}
    cost_pre = {start: 0} #keeps track of values before adding heuristic cost
    while frontier:
        curr = frontier.pop()
        curr = curr[1] #only need node
        if curr not in explored:
            explored.add(curr) #add curr state to explored
            
            if curr == goal: #terminating condition
                return path[curr]
            for i in sorted(graph.neighbors(curr), key = lambda x:graph.get_edge_weight(curr, x) + heuristic(graph, x, goal)):
                newCost = cost_pre[curr] + graph.get_edge_weight(curr, i)
                if i not in cost or newCost + heuristic(graph, i, goal) < cost[i]:
                    cost_pre[i] = newCost
                    cost[i] = newCost + heuristic(graph, i, goal)
                    frontier.append((cost[i], i))
                    path[i] = path[curr] + [i]
    return None


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    #TODO: finish this function!
    #raise NotImplementedError 
    if start == goal:
        return []
    frontier1 = PriorityQueue() #initialize start frontier
    frontier2 = PriorityQueue() #initialize goal frontier
    frontier1.append((0, start))
    frontier2.append((0, goal))
    explored1 = set()
    explored2 = set()
    path1 = {start: [start]} #dict to keep track of path
    path2 = {goal: [goal]}
    cost1 = {start: 0} #dict to keep track of cost
    cost2 = {goal: 0}
    maxBound = float('inf')
    intersect = set()
    while frontier1 and frontier2:
        curr = frontier1.pop()
        currCost = curr[0] 
        curr = curr[1]

        if curr not in explored1:
            explored1.add(curr) #add curr state to explored
            for i in sorted(graph.neighbors(curr), key=lambda x:graph.get_edge_weight(curr, x)):
                newCost = currCost + graph.get_edge_weight(curr, i)
                if i in explored2:
                    intersect.add((newCost + cost2[i], i))
                    maxBound = min(maxBound, newCost + cost2[i])
                if i not in cost1 or newCost < cost1[i]:
                    cost1[i] = newCost
                    path1[i] = path1[curr] + [i]
                    frontier1.append((newCost, i))
        curr = frontier2.pop()
        currCost = curr[0] 
        curr = curr[1]
     
        if curr not in explored2:
            explored2.add(curr) #add curr state to explored
            for i in sorted(graph.neighbors(curr), key=lambda x:graph.get_edge_weight(curr, x)):
                newCost = currCost + graph.get_edge_weight(curr, i)
                if i in explored1:
                    intersect.add((newCost + cost1[i], i))
                    maxBound = min(maxBound, newCost + cost1[i])
                if i not in cost2 or newCost < cost2[i]:
                    cost2[i] = newCost
                    path2[i] = path2[curr] + [i]
                    frontier2.append((newCost, i))

        if maxBound < frontier1.top()[0] + frontier2.top()[0]:
            tmp = sorted(intersect)[0][1]
            return path1[tmp][:-1] + path2[tmp][::-1]
    return None


def bidirectional_a_star(graph, start, goal,
                         heuristic=null_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #raise NotImplementedError
    if start == goal:
        return []
    frontier1 = PriorityQueue() #initialize start frontier
    frontier2 = PriorityQueue() #initialize goal frontier
    frontier1.append((0, start))
    frontier2.append((0, goal))
    explored1 = set()
    explored2 = set()
    path1 = {start: [start]} #dict to keep track of path
    path2 = {goal: [goal]}
    cost1 = {start: 0} #dict to keep track of cost
    cost2 = {goal: 0}
    cost1_pre = {start: 0} #dict to keep track of cost before adding heuristic
    cost2_pre = {goal: 0}
    intersect = set()
    maxBound = float('inf')
    while frontier1 and frontier2:
        curr = frontier1.pop()
        curr = curr[1]

        if heuristic == euclidean_dist_heuristic:
            if curr in explored2:
                intersect.add((cost1_pre[curr] + cost2[curr], curr))
                maxBound = min(maxBound, cost1_pre[curr] + cost2[curr])

        if curr not in explored1:
            explored1.add(curr) #add curr state to explored

            for i in sorted(graph.neighbors(curr), key=lambda x:graph.get_edge_weight(curr, x) + heuristic(graph, x, goal)):
                newCost = cost1_pre[curr] + graph.get_edge_weight(curr, i) 
                if heuristic == null_heuristic:
                    if i in explored2:
                        intersect.add((newCost + cost2[i], i))
                        maxBound = min(maxBound, newCost + cost2[i])
                if i not in cost1 or newCost + heuristic(graph, i, goal) < cost1[i]:
                    cost1_pre[i] = newCost
                    cost1[i] = newCost + heuristic(graph, i, goal)
                    path1[i] = path1[curr] + [i]
                    frontier1.append((cost1[i], i))
     
        curr = frontier2.pop()
        curr = curr[1]
    
        if heuristic != null_heuristic:
            if curr in explored1: #terminating condition
                intersect.add((cost2_pre[curr] + cost1[curr], curr))
                maxBound = min(maxBound, cost2_pre[curr] + cost1[curr])

        if curr not in explored2:
            explored2.add(curr) #add curr state to explored

            for i in sorted(graph.neighbors(curr), key=lambda x:graph.get_edge_weight(curr, x)):
                newCost = cost2_pre[curr] + graph.get_edge_weight(curr, i) 
                if heuristic == null_heuristic:
                    if i in explored1:
                        intersect.add((newCost + cost1[i], i))
                        maxBound = min(maxBound, newCost + cost1[i])
                if i not in cost2 or newCost + heuristic(graph, i, start) < cost2[i]:
                    cost2_pre[i] = newCost
                    cost2[i] = newCost + heuristic(graph, i, start)
                    path2[i] = path2[curr] + [i]
                    frontier2.append((cost2[i], i))

        if maxBound < frontier1.top()[0] + frontier2.top()[0]:
            tmp = sorted(intersect)[0][1]
            return path1[tmp][:-1] + path2[tmp][::-1]
    return None


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    #raise NotImplementedError
    return "Jamie Kim"


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
