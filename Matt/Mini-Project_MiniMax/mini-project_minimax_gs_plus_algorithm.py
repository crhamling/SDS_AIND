#
# Mini Project : MiniMax
#

from copy import deepcopy

# board dimensions
xlim, ylim = 3, 2  


#
# Class GameState
#

class GameState:
    """
    Attributes
    ----------
    _board: list(list)
        Represent the board with a 2d array _board[x][y]
        where open spaces are 0 and closed spaces are 1
    
    _parity: bool
        Keep track of active player initiative (which
        player has control to move) where 0 indicates that
        player one has initiative and 1 indicates player 2
    
    _player_locations: list(tuple)
        Keep track of the current location of each player
        on the board where position is encoded by the
        board indices of their last move, e.g., [(0, 0), (1, 0)]
        means player 1 is at (0, 0) and player 2 is at (1, 0)
    """

    #
    # Initialize a New Board and Set Attributes per the above Requirments
    #
    def __init__(self):
        self._board = [[0] * ylim for _ in range(xlim)] # Create a Blank Board using a "List Generator Comprehenion"
        self._board[-1][-1] = 1                         # block lower-right corner
        self._parity = 0                                # Player One has first Turn
        self._player_locations = [None, None]           # No Players are on the Board

    #
    # forecast_move
    #
    def forecast_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state. 
        
        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
        """
        
        if move not in self.get_legal_moves():
            raise RuntimeError("Attempted forecast of illegal move")  # Verify the next move is a legal one, raise error for illegal moves
        newBoard = deepcopy(self)                                     # Create a New Board from the current board         
        newBoard._board[move[0]][move[1]] = 1                         # Declare the target position for active players next move CLOSED
        newBoard._player_locations[self._parity] = move               # Set the player LOCATION of whose TURN it is to the MOVE (target position)
        newBoard._parity ^= 1                                         # Give the next move to the opponent
        
        return newBoard 

    #
    # get_legal_moves
    #
    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player.  Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.)
        """
        loc = self._player_locations[self._parity]                   # Determine the Location of the current move player
        if not loc:
            return self._get_blank_spaces()                          # In the case it's the first time, return blank spaces for board
        moves = []                                                   # Create a List to hold the MOVES
        rays = [(1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1)]                    # Rays = all permutations of the tuple
                
        # Return Legal Moves subject to board limitations        
        for dx, dy in rays:
            _x, _y = loc
            while 0 <= _x + dx < xlim and 0 <= _y + dy < ylim:
                _x, _y = _x + dx, _y + dy
                if self._board[_x][_y]:
                    break
                moves.append((_x, _y))
                
        return moves

    #
    # get_blank_spaces
    #
    def _get_blank_spaces(self):
        """ Return a list of blank spaces on the board."""
        return [(x, y) for y in range(ylim) for x in range(xlim)
                if self._board[x][y] == 0]




#
# MiniMax Helper Functions
#


#
# terminal_test
#
def terminal_test(gameState):
    """ Return True if the game is over for the active player
    and False otherwise.
    """
    return not bool(gameState.get_legal_moves())            # by Assumption 1 : Return false if there are no more legal moves left.


#
# min_value
#
def min_value(gameState):
    """ Return the value for a win (+1) if the game is over,
    otherwise return the minimum value over all legal child
    nodes.
    """
    if terminal_test(gameState):
        return 1                                                # by Assumption 2 : Signal the game has ended.
    v = float("inf")                                            # float("inf") : Pos Infinity - acts as a "unbounded" UPPER value for comparison
    for m in gameState.get_legal_moves():                       # For each legal move in the gamestate (assuming the game is still ongoing)
        v = min(v, max_value(gameState.forecast_move(m)))       # set V to the Min of (Infinity and the Max_Value of ALL the forecasted next (legal) moves)
    return v

#
# max_value
#
def max_value(gameState):
    """ Return the value for a loss (-1) if the game is over,
    otherwise return the maximum value over all legal child
    nodes.
    """
    if terminal_test(gameState):
        return -1                                               # by Assumption 2 : Signal the game has ended.
    v = float("-inf")                                           # float("-inf") : Neg Infinity - acts as a "unbounded" LOWER value for comparison
    for m in gameState.get_legal_moves():                       # For each legal move in the gamestate (assuming the game is still ongoing)
        v = max(v, min_value(gameState.forecast_move(m)))       # set V to the Min of (Infinity and the Max_Value of ALL the forecasted next (legal) moves)
    return v




##
## Test Harnass
##


# Create an Empty Board (Initialize) 
g = GameState()

# Check Board : [[0, 0], [0, 0], [0, 1]]
g._board

# Check Parity (those turn) : 0
g._parity

# Check Player Locations : [None, None]
g._player_locations

# Check the Openness or Closedness of various spaces on the board
g._board[0][0]  # Top Left Hand Space
g._board[2][1]  # Bottom Right Hand Space (Blocked at the start)

# Get the legal moves for Player 1
p1_empty_moves = g.get_legal_moves()

# Check what the Legal Moves actually are : [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
p1_empty_moves

# Check it's length : 5
len(p1_empty_moves)

#
# Player ONE First Move (0,0)
#

# Execute a Move: (0, 0) => Return a new Board Configuration, save it.
g1 = g.forecast_move((0, 0))


# Test and Illegal Move => Expect a Runtime Error to be raised
# g1.forecast_move((2, 1))

#
# Check New Board Configuration After the First Move (Player 1)
#

# Check Board : [[1, 0], [0, 0], [0, 1]]
g1._board

# Check Parity : 1
g1._parity

# Check Player Locations : [(0, 0), None]
g1._player_locations

# What are the legal moves at this point? : [(1, 0), (2, 0), (0, 1), (1, 1)]
g1.get_legal_moves()

# How many legal moves are there? : 4 => Notice that (0,0) is now absent from the list of legal moves.
len(g1.get_legal_moves())


# Get Legal Moves for Player 2 (their first move) given the last move of player 1
p2_empty_moves = g1.get_legal_moves()

# Check P2 Move Options (First Move) : [(1, 0), (2, 0), (0, 1), (1, 1)]
p2_empty_moves

# Check it's length : 4
len(p2_empty_moves)

#
# Player TWO First Move (2,0)
#

# Execute a Move: (2, 0) => Return a new Board Configuration from g1
g2 = g1.forecast_move((2, 0))

# Check New Board Configuration After the First Move ofPlayer 1 and the first move of player 2 : [[1, 0], [0, 0], [1, 1]]
# Note - p1 @ TLHS, and p2 @ TRHS are both occupied, plus the starting space of [2,1]
g2._board

# Check Parity : 0 
g2._parity

# Check Player Locations : [(0, 0), (2, 0)]
# Note - p1 @ TLHS, and p2 @ TRHS.
g2._player_locations


#
# Player ONE SECOND Move
#

# Game Over? : False
terminal_test(g2)

# P1 2nd Legal Moves:
p1_2nd_moves = g2.get_legal_moves()

# P1 2nd Move:
g3 = g2.forecast_move((1, 0))
g3._board

# Game Over? : False
terminal_test(g3)
min_value(g3) # Returns WIN value +1 if game is over
max_value(g3) # Returns LOSS value -1 if game is over

# Legal Moves : [(1, 1)]
g3.get_legal_moves()

# Player Locations : [(1, 0), (2, 0)]
g3._player_locations


#
# Player TWO SECOND Move
#

# P2 2nd Legal Moves: [(1, 1)]
p2_2nd_moves = g3.get_legal_moves()

# P2 2nd Move:
g4 = g3.forecast_move((1, 1))
g4._board

# Game Over? : False
terminal_test(g4)
min_value(g4) # Returns WIN value +1 if game is over, else max value over all child nodes
max_value(g4) # Returns LOSS value -1 if game is over, else min value over all child nodes



#
# Player ONE THIRD Move
#


# P1 3rd Legal Moves: [(0, 1)]
p1_3rd_moves = g4.get_legal_moves()


# P1 3rd Move:
g5 = g4.forecast_move((0, 1))
g5._board


# Game Over? : TRUE
terminal_test(g5)
min_value(g5) # Returns WIN value +1 if game is over, else max value over all child nodes : 1
max_value(g5) # Returns LOSS value -1 if game is over, else min value over all child nodes : -1





# 
# Test from Game State Code
#

print("Creating empty game board...")
g = GameState()

print("Getting legal moves for player 1...")
p1_empty_moves = g.get_legal_moves()
print("Found {} legal moves.".format(len(p1_empty_moves or [])))

print("Applying move (0, 0) for player 1...")
g1 = g.forecast_move((0, 0))

print("Getting legal moves for player 2...")
p2_empty_moves = g1.get_legal_moves()
if (0, 0) in set(p2_empty_moves):
    print("Failed\n  Uh oh! (0, 0) was not blocked properly when " +
          "player 1 moved there.")
else:
    print("Everything looks good!")
    



#
# Test from MiniMax Algorithm 
#

g = GameState()

print("Calling min_value on an empty board...")
v = min_value(g)

if v == -1:
    print("min_value() returned the expected score!")
else:
    print("Uh oh! min_value() did not return the expected score.")


















