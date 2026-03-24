# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack

    # Stack for DFS
    stack = Stack()

    # Each element = (state, path_to_state)
    start = problem.getStartState()
    stack.push((start, []))

    # Visited set to avoid revisiting nodes
    visited = set()

    while not stack.isEmpty():

        state, path = stack.pop()

        # If already visited → skip
        if state in visited:
            continue

        # Mark as visited
        visited.add(state)

        # Check goal
        if problem.isGoalState(state):
            return path

        # Expand successors
        for successor, action, cost in problem.getSuccessors(state):

            if successor not in visited:
                new_path = path + [action]
                stack.push((successor, new_path))

    # If no solution found
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    from util import Queue

    # Queue for BFS
    queue = Queue()

    # Each element = (state, path_to_state)
    start = problem.getStartState()
    queue.push((start, []))

    # Visited set
    visited = set()
    visited.add(start)

    while not queue.isEmpty():

        state, path = queue.pop()

        # Goal check
        if problem.isGoalState(state):
            return path

        # Expand successors
        for successor, action, cost in problem.getSuccessors(state):

            if successor not in visited:
                visited.add(successor)
                new_path = path + [action]
                queue.push((successor, new_path))

    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""

    from util import PriorityQueue

    # Priority Queue ordered by total cost
    pq = PriorityQueue()

    start = problem.getStartState()

    # Each element = (state, path, cost_so_far)
    pq.push((start, [], 0), 0)

    # Dictionary to track lowest cost to each state
    visited = {}

    while not pq.isEmpty():

        state, path, cost = pq.pop()

        # If visited with lower cost before → skip
        if state in visited and visited[state] <= cost:
            continue

        # Record best cost
        visited[state] = cost

        # Goal check
        if problem.isGoalState(state):
            return path

        # Expand successors
        for successor, action, stepCost in problem.getSuccessors(state):

            new_cost = cost + stepCost
            new_path = path + [action]

            pq.push((successor, new_path, new_cost), new_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    from util import PriorityQueue

    pq = PriorityQueue()
    start = problem.getStartState()

    # (state, path, cost_so_far)
    pq.push((start, [], 0), heuristic(start, problem))

    # best cost to each state
    visited = {}

    while not pq.isEmpty():

        state, path, cost = pq.pop()

        # Skip if we already found a cheaper path
        if state in visited and visited[state] <= cost:
            continue

        visited[state] = cost

        # Goal check
        if problem.isGoalState(state):
            return path

        # Expand successors
        for successor, action, stepCost in problem.getSuccessors(state):

            new_cost = cost + stepCost
            new_path = path + [action]

            priority = new_cost + heuristic(successor, problem)

            pq.push((successor, new_path, new_cost), priority)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
