# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  print("SOUTH: ",s, "WEST: ",w)
  return  [s,s,w,s,w,w,s,w]



def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()





def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  
  """
  "*** YOUR CODE HERE ***"
  # print "Start:", problem.getStartState()
  # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  # print "Start's successors:", problem.getSuccessors(problem.getStartState())
  # print "Type Starts Sccessors:", type(problem.getSuccessors(problem.getStartState()))
  # print "TYPE: ", type(problem)
  # print "DIR: ", dir(problem)
  # print "VARS: ", vars(problem)
  # print "DICT: ", problem.__dict__
  # print "GOAL: ", problem.goal
  # print "_visitedlist: ", problem._visitedlist
  # print "_expanded: ", problem._expanded
  # print "_visited: ", problem._visited
  # print "startState: ", problem.startState
  # print "TYPE-startState: ", type(problem.startState)
  # print "DOC: ", problem.__doc__
  # print "INIT: ", problem.__init__
  # print "MODULE: ", problem.__module__
  # print "COSTFN: ", problem.costFn
  # print "Next successors:", problem.getSuccessors((5,4))
  # print "\n****************"
  # print "*** Start BFS **"
  # print "****************\n"
  
  ss_node = [problem.startState, 'NA', 0]          # Experiment - full node w/start state
  frontier = [[get_position(ss_node)]]             # populate the Frontier w/the start state position
  frontier_full_node = [[ss_node]]                 # populate full node frontier w/full node of start position
  explored = []                                    # initialize explored state
  path = []                                        # initialize Path
  path_full_node = []                              # intiialize full node path
  node = []                                        # initialize node
  node_full_node = []                              # initialize full node 
  new_path_full_node = []                          # initialize new_path_full_node
  
  # keeps looping until all possible paths have been checked
  while frontier:
      # Check for empty frontier - no solution possible.
      if not frontier:
          raise Exception('No Solution Possible')
      # pop the first path from the frontier & full node frontier
      path = frontier.pop(0)
      path_full_node = frontier_full_node.pop(0)
      # get the last node from the path & full node path
      node = path[-1]
      node_full_node = path_full_node[-1]
      # Explore if node not yet explored
      if node not in explored:
          neighbours = problem.getSuccessors(node)
          # go through all neighbour nodes, construct a new path and push it into the frontier
          # (and full node frontier),along with their full node doppelgangers
          for neighbor in neighbours:
              new_path = list(path)
              new_path_full_node = list(path_full_node)
              new_path.append(neighbor[0])
              new_path_full_node.append(neighbor)
              frontier.append(new_path)
              frontier_full_node.append(new_path_full_node)
              # if you find the goal - return the path (and full node path)
              if neighbor[0] == problem.goal:
                  # print "\n** PATH ** \n", new_path, extract_sequence_of_moves(new_path_full_node)
                  # return the extracted sequence of moves to reach the goal per the utility function.
                  return extract_sequence_of_moves(new_path_full_node)
              # mark node as explored if it's not been explored before.
              if node not in explored:
                explored.append(node)
  # if we haven't returned prior, return now.                
  return extract_sequence_of_moves(new_path_full_node)
  
  
def extract_sequence_of_moves(path):
    sequence_of_moves = []
    new_path = path[1:]
    for node in new_path:
        sequence_of_moves.append(node[1])
    return sequence_of_moves
        
def get_position(node):
    return node[0]
    
def get_direction(node):
    return node[1]

  
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
