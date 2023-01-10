import random


# Make random move
def random_player(game):
    moves = game.legal_moves()
    return random.choice(moves)