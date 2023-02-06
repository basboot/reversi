import random
import time

from random_player import random_player
from nim import *

from alpha_zero import *
from alpha_reversi import *


if __name__ == '__main__':

    print("AlphaReversi")
    game = AlphaReversi()
    alpha_zero = AlphaZero(game)
    #
    # move = alpha_zero.player(game)
    #
    # print(move)

    def alpha_reversi_player(game):
        return alpha_zero.player(game, prediction_only=False, mcts_only=False, rollout=False,
                                 always_renew_mcts=True, nn_compete=True, n_games=25,
                                 n_simulations=100, n_samples=100, n_validation=0)


    game = None

    tie = 0
    player0 = 0
    player1 = 0


    # Let op player 0 en 1 / maar WHITE = 1 en BLACK = 2! (zwart begint)
    players = [random_player, alpha_reversi_player]
    # players = [alpha_zero.player, alpha_zero.player]
    points = [0, 0]

    start = time.time()
    for i in range(1000):
        print("Game", i)
        game = AlphaReversi()

        # select player that plays with BLACK (= starting player)
        first_player = random.randint(0, 1)

        while True:
            print(".", end='')
            moves = game.legal_moves()
            if len(moves) == 0:
                # game over
                break

            current_player = (game.current_player + first_player) % 2
            move = players[current_player](game)
            game = game.perform_move(move)


        winning_player, _, _ = game.winning_player()
        if winning_player == 0:
            tie += 1
        else:
            points[(winning_player + first_player) % 2] += 1

        # save nn after each game
        alpha_zero.save_nn_values()

    print("P0 - P1", points[0], points[1])

    end = time.time()
    print("Time consumed in working: ", end - start)