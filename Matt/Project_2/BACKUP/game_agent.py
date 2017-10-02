"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


import random
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


#
# MATCH RESULTS
#


#                         *************************
#                              Playing Matches
#                         *************************
#
#  Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3
#                         Won | Lost   Won | Lost   Won | Lost   Won | Lost
#     1       Random       7  |   3     9  |   1     9  |   1     8  |   2
#     2       MM_Open      8  |   2     8  |   2     7  |   3     6  |   4
#     3      MM_Center     7  |   3     6  |   4     6  |   4     6  |   4
#     4     MM_Improved    8  |   2     6  |   4     4  |   6     6  |   4
#     5       AB_Open      5  |   5     4  |   6     5  |   5     5  |   5
#     6      AB_Center     7  |   3     5  |   5     7  |   3     3  |   7
#     7     AB_Improved    4  |   6     4  |   6     4  |   6     6  |   4
# --------------------------------------------------------------------------
#            Win Rate:      65.7%        60.0%        60.0%        57.1%





#
# HEURISTIC heuristic_player_moves_vs_player_distance_to_center
#
def heuristic_player_moves_vs_player_distance_to_center(game, player):
    """Outputs a score equal to the difference in the number of moves available to the
    two players multiplied by the active players distance to the center of the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Get the num of moves left for both players and take the difference
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    moves_diff = own_moves - opp_moves
    
    # Get the location of the player relative to the center of the board
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    dist_to_center = ((h - y)**2 + (w - x)**2)
    
    return float(moves_diff * dist_to_center)


#
# HEURISTIC heuristic_player_opponent_vs_center_distance
#
def heuristic_player_opponent_vs_center_distance(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board relative to the difference of the position of the player vs opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    # Get the dimmensions of the Board
    w, h = game.width / 2., game.height / 2.
    
    # Get Player Location
    py, px = game.get_player_location(player)
    
    # Get Opponents location
    oy, ox = game.get_player_location(game.get_opponent(player))
    
    # Return the heuristic calculation
    return float( (h - abs(py -oy))**2 + (w - abs(px - ox))**2 )


#
# HEURISTIC heuristic_square_difference_available_moves
#
def heuristic_square_difference_available_moves(game, player):
    """Outputs a score equal to the square of the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float((own_moves - opp_moves)*2)
    

#
# COSTOM SCORE
#
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic_square_difference_available_moves(game, player)
    
    
#
# COSTOM SCORE_2
#    
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic_player_moves_vs_player_distance_to_center(game, player)


#
# COSTOM SCORE_3
#
def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic_player_opponent_vs_center_distance(game, player)




class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


#
# CLASS MinimaxPlayer
#
class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    #
    # FUNCTION: minimax
    #
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        # For Readability Purposes
        NEG_INFINITY = float('-inf')
        
        # Check to ensure agent handles timeout
        check_timer(self)

        # Using Anomyous FN's and MAX to determine the move along a branch of the game tree 
        # that has the best possible value. 
        #
        # Note - this lambda worked for the submission, but not the tournament.
        #
        # return max(game.get_legal_moves(),
        #    key=lambda m: self.min_value(game.forecast_move(m), depth))
        

        # Keep track of the best score, initialized low to track the maximum value
        best_score = NEG_INFINITY
        
        # Initialize the best_move variable to something reasonable
        best_move = (-1, -1)
        
        # For each possible legal move
        for m in game.get_legal_moves():
            
            # Set v to min_value of all possible moves in the gamestate, Increment depth counter
            v = self.min_value(game.forecast_move(m), depth)

            # If an overall high score is found, keep it
            if v > best_score:
                best_score = v
                best_move = m
                
        return best_move        
        
        
    #
    # FUNCTION: max_value
    #
    def max_value(self, game, depth):
        
        # For Readability Purposes
        LEAF_NODE = 1
        NEG_INFINITY = float('-inf')
        
        check_timer(self) # Timeout Checker

        # Evaluate the heuristic at the leaf nodes
        if depth == LEAF_NODE:
            return self.score(game, self)

        # Initialize the MAX tracker to Negative infinity
        best_value = NEG_INFINITY
        
        # For each possible legal move
        for m in game.get_legal_moves():
            
            # Set v to min_value of all possible moves in the gamestate, Increment depth counter
            v = self.min_value(game.forecast_move(m), depth-1)
            
            # If an overall high score is found, keep it
            if v > best_value:
                best_value = v
                
        return best_value
        
        
    #
    # FUNCTION: min_value
    #
    def min_value(self, game, depth):
        
        # For Readability Purposes
        LEAF_NODE = 1
        POS_INFINITY = float('inf')
        
        check_timer(self) # Timeout Checker
        
        # Evaluate the heuristic at the leaf nodes
        if depth == LEAF_NODE:
            return self.score(game, self)

        # Initialize the MIN tracker to Positive infinity
        best_value = POS_INFINITY
        
        # For each legal move
        for m in game.get_legal_moves():
            
            # Set v to max_value of all possible moves in the gamestate, Increment depth counter
            v = self.max_value(game.forecast_move(m), depth-1)
            
            # If an overall low score is found, keep it
            if v < best_value:
                best_value = v
                
        return best_value
 

#
# CLASS AlphaBetaPlayer
#
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        # Timeouts will cause the FN to fail so we must initalize the best_move 
        # variable to something reasonable
        best_move = (-1, -1)

        # Keep track of the current depth
        depth = 1

        try:
            # Catch the timer expiration exceptions
            while True:
                best_move = self.alphabeta(game, depth) # Capture BM for level
                depth += 1                              # ++depth
        except SearchTimeout:
            return best_move
            
        return best_move

    #
    # FUNCTION alphabeta
    # 
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        # For Readability Purposes
        NEG_INFINITY = float('-inf')
        
        # Check to ensure agent handles timeout
        check_timer(self)

        # Initialize the MAX tracker to Negative infinity
        best_score = NEG_INFINITY
        
        # Initialize the best_move variable to something reasonable
        best_move = (-1, -1)
        
        # For each legal move
        for m in game.get_legal_moves():
            
            # Set v to min_value of all possible a/b pruned moves in the gamestate
            v = self.min_value(game.forecast_move(m), depth, alpha, beta)
            
            # If an overall high score is found, keep it
            if v > best_score:
                best_score = v
                best_move = m
                
            alpha = max(alpha, best_score)  # set alpha to max of alpha and best_score
            
        return best_move 

        
        
    #
    # FUNCTION mav_value
    #
    def max_value(self, game, depth, alpha, beta):
        
        # For Readability Purposes
        LEAF_NODE = 1
        NEG_INFINITY = float('-inf')
        
        # Check to ensure agent handles timeout
        check_timer(self)

        # Evaluate the heuristic at the leaf nodes
        if depth == LEAF_NODE:
            return self.score(game, self)

        # Initialize the MAX tracker to Negative infinity
        best_value = NEG_INFINITY
        
        # For each legal move
        for m in game.get_legal_moves():
            
            # Set v to min_value of all possible moves in the gamestate, update key attributes
            v = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            
            if v > best_value:
                best_value = v               # If an overall High score is found, keep it
                
            if best_value >= beta:
                return best_value            # Prune the search space
                
            alpha = max(alpha, best_value)   # update ALPHA
            
        return best_value
        
        
    #
    # FUNCTION min_value
    #
    def min_value(self, game, depth, alpha, beta):
        
        # For Readability Purposes
        LEAF_NODE = 1
        POS_INFINITY = float('inf')
        
        # Check to ensure agent handles timeout
        check_timer(self)

        # Evaluate the heuristic at the leaf nodes
        if depth == LEAF_NODE:
            return self.score(game, self)

        # Initialize the MIN tracker to Positive infinity
        best_value = POS_INFINITY
        
        # For each legal move
        for m in game.get_legal_moves():
            
            # Set v to max_value of all possible moves in the gamestate, update key attributes
            v = self.max_value(game.forecast_move(m), depth-1, alpha, beta)

            if v < best_value:
                best_value = v             # If an overall High score is found, keep it

            if best_value <= alpha:
                return best_value          # Prune the search space
                
            beta = min(beta, best_value)   # Update BETA
            
        return best_value

        

#
# Utility FN to check for timer issues
#
def check_timer(self):
    # Check to ensure agent handles timeout
    if self.time_left() < self.TIMER_THRESHOLD:
        raise SearchTimeout()
        
        
        
        
        
        
        
        
        
        
        
        