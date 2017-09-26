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
            raise RuntimeError("Attempted forecast of illegal move")  # Verify the enxt move is a legal one
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





##
## Test Harnass
##


# Create an Empty Board 
g = GameState()

# Check Board
g._board

# Check Parity
g._parity

# Check Player Locations
g._player_locations

# Check the Openness or Closedness of various spaces on the board
g._board[0][0]  # Top Left Hand Space
g._board[2][1]  # Bottom Right Hand Space (Blocked at the start)

# Get the legal moves for Player 1
p1_empty_moves = g.get_legal_moves()
print("Found {} legal moves.".format(len(p1_empty_moves or [])))

# Check Legal Moves
p1_empty_moves

len(p1_empty_moves)

# Execute a Move: (0, 0) => Return a new Board Configuration
print("Applying move (0, 0) for player 1...")
g1 = g.forecast_move((0, 0))


# Check New Board Configuration
g1

# Check Board
g1._board

# Check Parity
g1._parity

# Check Player Locations
g1._player_locations


# Get Legal Moves for Player 2
p2_empty_moves = g1.get_legal_moves()

# Check Those Moves
p2_empty_moves


if (0, 0) in set(p2_empty_moves):
    print("Failed\n  Uh oh! (0, 0) was not blocked properly when " +
          "player 1 moved there.")
else:
    print("Everything looks good!")
    




















































