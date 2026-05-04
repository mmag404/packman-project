# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        from util import manhattanDistance

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        foodList = newFood.asList()

        # 🍎 Food score
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            foodScore = 1 / (minFoodDist + 1)
        else:
            foodScore = 0

        # 👻 Ghost score
        ghostScore = 0
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)

            if ghost.scaredTimer > 0:
                ghostScore += 2 / (dist + 1)
            else:
                if dist < 2:
                    return -float('inf')
                ghostScore -= 2 / (dist + 1)

        return successorGameState.getScore() + foodScore + ghostScore
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):

        def minimax(agentIndex, depth, state):
            # Stop condition
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # PACMAN (MAX)
            if agentIndex == 0:
                best = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(1, depth, successor)
                    best = max(best, val)
                return best

            # GHOSTS (MIN)
            else:
                best = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)

                    # If last ghost → go to Pacman and increase depth
                    if agentIndex == numAgents - 1:
                        val = minimax(0, depth + 1, successor)
                    else:
                        val = minimax(agentIndex + 1, depth, successor)

                    best = min(best, val)
                return best

        # ROOT: choose best action for Pacman
        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def alpha_beta(state, depth, agentIndex, alpha, beta):
            # Terminal condition
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # PACMAN (MAX)
            if agentIndex == 0:
                value = float('-inf')
                actions = state.getLegalActions(agentIndex)

                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alpha_beta(successor, depth, 1, alpha, beta))

                    if value > beta:   # prune
                        return value

                    alpha = max(alpha, value)

                return value

            # GHOSTS (MIN)
            else:
                value = float('inf')
                actions = state.getLegalActions(agentIndex)

                nextAgent = agentIndex + 1
                nextDepth = depth

                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1

                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alpha_beta(successor, nextDepth, nextAgent, alpha, beta))

                    if value < alpha:   # prune
                        return value

                    beta = min(beta, value)

                return value

        # ROOT (Pacman chooses action)
        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alpha_beta(successor, 0, 1, alpha, beta)

            if value > bestScore:
                bestScore = value
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def expectimax(state, depth, agentIndex):

            # Terminal condition
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # PACMAN (MAX)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    value = max(value, expectimax(successor, depth, 1))
                return value

            # GHOST (EXPECTATION)
            else:
                actions = state.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return self.evaluationFunction(state)

                probability = 1 / len(actions)
                value = 0

                nextAgent = agentIndex + 1
                nextDepth = depth

                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value += probability * expectimax(successor, nextDepth, nextAgent)

                return value

        # Choose best action for Pacman
        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 0, 1)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
