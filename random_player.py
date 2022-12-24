import random


def random_player(game):
    moves = game.legal_moves()
    return random.choice(moves)