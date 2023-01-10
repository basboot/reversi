import math
from reversi import *

# Make move that results in most points
def greedy_player(game):
    moves = game.legal_moves()

    best_points = -math.inf
    best_move = None

    for move in moves:
        virtual_game = game.perform_move(move)
        _, white_points, black_points = game.winning_player()
        my_points = white_points if game.current_player == WHITE else black_points

        if my_points > best_points:
            best_move = move
            best_points = my_points

    return best_move
