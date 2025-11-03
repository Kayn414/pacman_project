from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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
    minimax agent 
  """
  def value(self, state, indx, depth):
      legalActions = state.getLegalActions(indx)
      # Check for terminal states or depth limit
      if state.isWin() or state.isLose() or not legalActions: 
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      numAgents = state.getNumAgents()
      scores = [self.value(state.generateSuccessor(indx, action),
                         (indx+1) % numAgents,
                         depth - int(indx == numAgents-1))
              for action in legalActions]
      return max(scores) if indx == self.index else min(scores)
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 


      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    agentIndex = self.index
    numAgents = gameState.getNumAgents()
    legalActions = gameState.getLegalActions(agentIndex)
    random.shuffle(legalActions) 

    f = max if self.index == 0 else min
    return f(legalActions, key=lambda x: self.value(gameState.generateSuccessor(agentIndex, x),
                                                    (agentIndex+1) % numAgents,
                                                    self.depth - int(agentIndex == numAgents-1)))
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    agentIndex = self.index
    numAgents = gameState.getNumAgents()
    return self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % numAgents, self.depth - int(agentIndex == numAgents - 1))



class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """


    def value(indx, depth, state):
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(indx)
      if not legalActions:
        return self.evaluationFunction(state)

      nextAgent = (indx + 1) % state.getNumAgents()
      nextDepth = depth + 1 if nextAgent == self.index else depth

      if indx == self.index:  # Pacman: maximize
        return max(
          value(nextAgent, nextDepth, state.generateSuccessor(indx, action))
          for action in legalActions
        )
      else:  # Ghost: take expected value (average)
        values = [
          value(nextAgent, nextDepth, state.generateSuccessor(indx, action))
          for action in legalActions
        ]
        return sum(values) / len(values)

    bestScore = float('-inf')
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(self.index):
      succ = gameState.generateSuccessor(self.index, action)
      score = value(1, 0, succ)
      if score > bestScore:
        bestScore = score
        bestAction = action
    return bestAction
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    def value(indx, depth, state):
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(indx)
      if not legalActions:
        return self.evaluationFunction(state)

      nextAgent = (indx + 1) % state.getNumAgents()
      nextDepth = depth + 1 if nextAgent == self.index else depth

      if indx == self.index:
        return max(
          value(nextAgent, nextDepth, state.generateSuccessor(indx, action))
          for action in legalActions
        )
      else:
        values = [
          value(nextAgent, nextDepth, state.generateSuccessor(indx, action))
          for action in legalActions
        ]
        return sum(values) / len(values)

    succState = gameState.generateSuccessor(self.index, action)
    return value(1, 0, succState)



class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Biased-expectimax agent 
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    def biased_expectimax(indx, depth, state):
      if state.isWin() or state.isLose() or (depth == 0 and indx == 0):
        return self.evaluationFunction(state)
      
      numAgents = state.getNumAgents()
      legalActions = state.getLegalActions(indx)

      if len(legalActions) == 0:
        return self.evaluationFunction(state)
    
      nextAgent = (indx + 1) % numAgents
      nextDepth = depth - 1 if nextAgent == 0 else depth

      if indx == 0:
        return max(biased_expectimax(nextAgent, nextDepth, state.generateSuccessor(indx, action)) for action in legalActions)
      else:
        lenAction = len(legalActions)
        bias = {}
        for act in legalActions:
          if act == Directions.STOP:
            bias[act] = 0.5 + (0.5 / lenAction) #more likely to choose stop
          else:
            bias[act] = 0.5 / (lenAction) # less likely to choose other actions
        expected_val = 0
        for act in legalActions:
          succ = state.generateSuccessor(indx, act)
          value = biased_expectimax(nextAgent, nextDepth, succ)
          expected_val += bias[act] * value
        return expected_val
      
    tmp_Score = float('-inf')
    tmp_Action = Directions.STOP
    for action in gameState.getLegalActions(self.index):
      succ = gameState.generateSuccessor(self.index, action)
      score = self.evaluationFunction(succ) if (succ.isWin() or succ.isLose()) else biased_expectimax(1, self.depth, succ)
      if score > tmp_Score:
        tmp_Score = score
        tmp_Action = action
    return tmp_Action
                                                                                                      
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    def biased_expectimax(indx, depth, state):
      if state.isWin() or state.isLose() or (depth == 0 and indx == 0):
        return self.evaluationFunction(state)
      
      numAgents = state.getNumAgents()
      legalActions = state.getLegalActions(indx)

      if len(legalActions) == 0:
        return self.evaluationFunction(state)
      
      nextAgent = (indx + 1) % numAgents
      nextDepth = depth - 1 if nextAgent == 0 else depth

      if indx == 0:
        return max(biased_expectimax(nextAgent, nextDepth, state.generateSuccessor(indx, action)) for action in legalActions)
      else:
        lenAction = len(legalActions)
        bias = {}
        for act in legalActions:
          if act == Directions.STOP:
            bias[act] = 0.5 + (0.5 / lenAction)
          else:
            bias[act] = 0.5 / (lenAction)
        expected_val = 0
        for act in legalActions:
          succ = state.generateSuccessor(indx, act)
          value = biased_expectimax(nextAgent, nextDepth, succ)
          expected_val += bias[act] * value
        return expected_val
      
    succState = gameState.generateSuccessor(self.index, action)
    return biased_expectimax(1, self.depth, succState)


class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Expectiminimax agent 
  """

  def Value(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)

    if gameState.isWin() or gameState.isLose() or not legalActions: return gameState.getScore()
    if depth == 0: return self.evaluationFunction(gameState)

    numAgents = gameState.getNumAgents()
    scores = [self.Value(gameState.generateSuccessor(agentIndex, action),
                         (agentIndex+1) % numAgents,
                         depth - int(agentIndex == numAgents-1))
              for action in legalActions]
    return max(scores) if agentIndex == 0 else \
           min(scores) if agentIndex % 2 == 1 else \
           sum(scores) / len(scores)

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """
    agentIndex = self.index
    numAgents = gameState.getNumAgents()

    legalActions = gameState.getLegalPacmanActions()
    random.shuffle(legalActions)

    f = max if self.index == 0 else min
    return legalActions[0] if agentIndex > 0 and agentIndex % 2 == 0 else \
           f(legalActions, key=lambda x: self.Value(gameState.generateSuccessor(agentIndex, x),
                                                    (agentIndex+1) % numAgents,
                                                    self.depth - int(agentIndex == numAgents-1)))

  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    agentIndex = self.index
    numAgents = gameState.getNumAgents()
    return self.Value(gameState.generateSuccessor(agentIndex, action),
                      (agentIndex+1) % numAgents,
                      self.depth - int(agentIndex == numAgents-1))


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """
  def value(self, state, indx, depth, alpha, beta):
    legalActions = state.getLegalActions(indx)

    if state.isWin() or state.isLose() or not legalActions: 
       return state.getScore()
    if depth == 0: 
      return self.evaluationFunction(state)
      
    numAgents = state.getNumAgents()
    if indx == 0 or indx % 2 == 1:
      value = float('-inf' if indx == 0 else 'inf')
      for action in legalActions:
        f = max if indx == 0 else min
        value = f(value, self.value(state.generateSuccessor(indx, action),
                                    (indx+1) % numAgents,
                                    depth - int(indx == numAgents-1),
                                    alpha, beta))
        if indx == 0: alpha = max(alpha, value)
        else: beta = min(beta, value)
        if beta < alpha: break
      return value
    else:
      scores = [self.value(state.generateSuccessor(indx, action),
                           (indx+1) % numAgents,
                           depth - int(indx == numAgents-1),
                           alpha, beta)
                for action in legalActions]
      return sum(scores) / len(scores)
  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    agentIndex = self.index
    numAgents = gameState.getNumAgents()

    legalActions = gameState.getLegalPacmanActions()
    random.shuffle(legalActions)

    f = max if self.index == 0 else min
    return legalActions[0] if agentIndex > 0 and agentIndex % 2 == 0 else \
           f(legalActions, key=lambda x: self.value(gameState.generateSuccessor(agentIndex, x),
                                                    (agentIndex+1) % numAgents,
                                                    self.depth - int(agentIndex == numAgents-1),
                                                    float('-inf'), float('inf')))
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    agentIndex = self.index
    numAgents = gameState.getNumAgents()
    return self.value(gameState.generateSuccessor(agentIndex, action),
                      (agentIndex+1) % numAgents,
                      self.depth - int(agentIndex == numAgents-1),
                      float('-inf'), float('inf'))


def betterEvaluationFunction(currentGameState):
  score = currentGameState.getScore()
  pacmanPos = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood().asList()
  ghostStates = currentGameState.getGhostStates()
  capsules = currentGameState.getCapsules()


  for ghost in ghostStates:
    ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
    if ghost.scaredTimer > 0:
      score += 200 / (ghostDist + 1)
    else:
      if ghostDist < 2:
        score -= 500
      else:
        score -= 2.0 / ghostDist + 1

  if foodList:
     minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
     score += 10 / minFoodDist
     score -= 4 * len(foodList)

  score -= 15 * len(capsules)
  if capsules:
    clostestCapusule = min(manhattanDistance(pacmanPos, capsule) for capsule in capsules)
    score += 5.0 / clostestCapusule 
  return score   

def choiceAgent():
  return 'ExpectimaxAgent'

# Abbreviation
better = betterEvaluationFunction
