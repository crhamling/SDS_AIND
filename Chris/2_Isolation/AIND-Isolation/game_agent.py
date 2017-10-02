"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    pass


def custom_score(game, player):
    
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    moves = len(game.get_legal_moves(player))
    oppMoves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(moves - oppMoves)


def custom_score_2(game, player):
    return custom_score(game, player)
    # TODO: finish this function!


def custom_score_3(game, player):
    return custom_score(game, player)
    # TODO: finish this function!



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


    def minimax(self, game, depth):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves_list = game.get_legal_moves(game.active_player)
        best_score = float("-inf")

        for m in legal_moves_list:
            val = self._min_max(game.forecast_move(m), depth-1, 'min')

            if val > best_score:
                best_score = val
                best_move = m
        
        return best_move
  
    
    def _min_max(self, game, depth, mode):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if mode == 'min':
            if game.get_legal_moves(game.active_player) == 0 or depth == 0:
                return self.score(game, game.inactive_player)
            best = float("inf")
            for m in game.get_legal_moves(game.active_player):
                best = min(best, self._min_max(game.forecast_move(m), depth-1, 'max'))

        elif mode == 'max':
            if game.get_legal_moves(game.active_player) == 0 or depth == 0:
                return self.score(game, game.active_player)
            best = float("-inf")
            for m in game.get_legal_moves(game.active_player):
                best = max(best, self._min_max(game.forecast_move(m), depth-1, 'min'))

        return best



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.bestInTime = (-1, -1)
        
        if(len(game.get_legal_moves(game.active_player))):
            self.bestInTime = game.get_legal_moves(game.active_player)[0]

        try:
            depth = 0
            while(True):
                self.bestInTime = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            return self.bestInTime  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return self.bestInTime

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves_list = game.get_legal_moves(game.active_player)
        best_score = float("-inf")

        for m in legal_moves_list:
            val = self._ab_min_max(game.forecast_move(m), depth-1, alpha, beta, 'min')

            if val > best_score:
                best_score = val
                self.bestInTime = m
            
            alpha = max(alpha, best_score)
            
        return self.bestInTime
  
    
    def _ab_min_max(self, game, depth, alpha, beta, mode):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        retVal = float("inf")
        for m in game.get_legal_moves(game.active_player):
            if mode == 'min':
                if game.get_legal_moves(game.active_player) == 0 or depth == 0:
                    return self.score(game, game.inactive_player)
                retVal = min(retVal, self._ab_min_max(game.forecast_move(m), depth-1, alpha, beta, 'max'))
                if retVal <= alpha:
                    return retVal
                beta = min(beta, retVal)
            elif mode == 'max':
                if game.get_legal_moves(game.active_player) == 0 or depth == 0:
                    return self.score(game, game.active_player)
                retVal = max(retVal, self._ab_min_max(game.forecast_move(m), depth-1, alpha, beta, 'min'))
                if retVal >= beta:
                    return retVal
                alpha = max(alpha, retVal)
        return retVal
