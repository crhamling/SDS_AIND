"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


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
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    #temporary heuristic, same as improved_score in sample_players.py
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def custom_score_2(game, player):
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    # TODO: finish this function!
    raise NotImplementedError


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
        legal_moves_list = game.get_legal_moves(game.active_player)
        best_move = (-1, -1)
        #initialize the best move to the first available legal moves to avoid forfeiting
        if len(legal_moves_list) != 0:
            best_move = legal_moves_list[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves_list = game.get_legal_moves(game.active_player)
        best_score = float("-inf")
        best_move = (-1, -1)
        #initialize the best move to the first available legal moves to avoid forfeiting
        if len(legal_moves_list) != 0:
            best_move = legal_moves_list[0]
        #loop through the available moves and apply minimax search to find best move
        for m in legal_moves_list:
            #min is called on first here, which will mutually recursively call on max until a score is reached for this move
            v = self.min_value(game.forecast_move(m), depth-1)
            #note that although min is called on first, the first comparison made will be a max comparison
            if v > best_score:
                best_score = v
                best_move = m
        
        return best_move

    """ minimax helper functions defined below 
    -------------------------------------"""
    
    def terminal_test(self, gameState, depth):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #Depth check. Prematurely stops the recursion, no need to look further because it will be too time consuming
        if depth == 0:
            return True
        #if there are no more legal moves, game over
        legal_moves = gameState.get_legal_moves(gameState.active_player)
        if len(legal_moves) == 0:
            return True
        else:
            return False
    
    
    def min_value(self, gameState, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #On your opponent's turn, he/she will try to minimize the score. 
        #This means opponent is active on min node (when min_value is called)
        #We check to see terminal test passes on opponents move, and give a score of positive because this outcome favors us.
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.inactive_player)
        
        v = float("inf")
        for m in gameState.get_legal_moves(gameState.active_player):
            v = min(v, self.max_value(gameState.forecast_move(m), depth-1))
        return v
    
    
    def max_value(self, gameState, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #On your turn, you wil try to maximize the score.
        #This means you are the active player on max node (when max_value is called)
        #We check to see if terminal test passes on your move, and give a score of negative because this outcome does not favor us.
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.active_player)
        
        v = float("-inf")  
        for m in gameState.get_legal_moves(gameState.active_player):
            v = max(v, self.min_value(gameState.forecast_move(m), depth-1))
        return v
    
    """ End of minimax helper functions
    -------------------------------------"""


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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves_list = game.get_legal_moves(game.active_player)
        best_move = (-1, -1)
        #initialize the best move to the first available legal moves to avoid forfeiting
        if len(legal_moves_list) != 0:
            best_move = legal_moves_list[0]
        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        try:
            #Continue to apply the search while increasing the depth until time runs out
            for deepening_depth in range(0, 9999):
                best_move = self.alphabeta(game, deepening_depth)
                
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves_list = game.get_legal_moves(game.active_player)
        best_score = float("-inf")
        best_move = (-1, -1)
        #initialize the best move to the first available legal moves to avoid forfeiting
        if len(legal_moves_list) != 0:
            best_move = legal_moves_list[0]
        #loop through the available moves and apply minimax alpha beta pruning search to find best move
        for m in legal_moves_list:
            #min is called on first here, which will mutually recursively call on max until a score is reached for this move
            v = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            #note that although min is called on first, the first comparison made will be a max comparison
            if v > best_score:
                best_score = v
                best_move = m
            #Note we do not need to consider pruning at this top level of the tree,
            #we just have to update alpha value of this maximizing node so lower nodes can use it for pruning.
            alpha = max(alpha, best_score)
        
        return best_move

    """ alphabeta helper functions defined below 
    -------------------------------------"""
    
    def terminal_test(self, gameState, depth):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #Depth check. Prematurely stops the recursion, no need to look further because it will be too time consuming
        if depth == 0:
            return True
        #if there are no more legal moves, game over
        legal_moves = gameState.get_legal_moves(gameState.active_player)
        if len(legal_moves) == 0:
            return True
        else:
            return False
    
    
    def min_value(self, gameState, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #On your opponent's turn, he/she will try to minimize the score. 
        #This means opponent is active on min node (when min_value is called)
        #We check to see terminal test passes on opponents move, and give a score of positive because this outcome favors us.
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.inactive_player)
        
        v = float("inf")
        for m in gameState.get_legal_moves(gameState.active_player):
            v = min(v, self.max_value(gameState.forecast_move(m), depth-1, alpha, beta))
            #pruning happens here when the current min value found is less than alpha
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    
    def max_value(self, gameState, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #On your turn, you wil try to maximize the score.
        #This means you are the active player on max node (when max_value is called)
        #We check to see if terminal test passes on your move, and give a score of negative because this outcome does not favor us.
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.active_player)
        
        v = float("-inf")  
        for m in gameState.get_legal_moves(gameState.active_player):
            v = max(v, self.min_value(gameState.forecast_move(m), depth-1, alpha, beta))
            #pruning happens here when the current max value found for this branch is greater than beta
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    """ End of alphabeta helper functions
    -------------------------------------"""
