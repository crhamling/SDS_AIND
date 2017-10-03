#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:01:11 2017

@author: anourkad
"""

#POSTERITY
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
    
    #this heuristic uses the avg number of legal moves of the next play if this play
    #is chosen minus the opponent's number of future legal moves
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    totalmoves = 0
    number_games = 0
    
    for m in game.get_legal_moves(player):
        new_game = game.forecast_move(m)
        totalmoves += len(new_game.get_legal_moves(player))
        number_games = number_games + 1
        
    Opptotalmoves = 0
    Oppnumber_games = 0
    
    for m in game.get_legal_moves(game.get_opponent(player)):
        Oppnew_game = game.forecast_move(m)
        Opptotalmoves += len(Oppnew_game.get_legal_moves(game.get_opponent(player)))
        Oppnumber_games = Oppnumber_games + 1
        
    heuristic = totalmoves/number_games
    OppHeuristic = Opptotalmoves/Oppnumber_games
    
    return float(heuristic - OppHeuristic)


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
    
    #blank spaces heuristic, subtract opp    
    #this player heuristic is based purely on chance. gambler's ai
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    return float(0)


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
    
    #this heuristic maximizes opponent's illegal moves
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(100 - opp_moves)