import random
import time

from random_player import random_player
from nim import *

from alpha_zero import *
from alpha_nim import *


if __name__ == '__main__':
    game = AlphaNim()
    alpha_zero = AlphaZero(game)
    #
    # move = alpha_zero.player(game)
    #
    # print(move)

    def alpha_nim_player(game):
        return alpha_zero.player(game, prediction_only=False, mcts_only=True, rollout=True)

    game = None

    tie = 0
    player0 = 0
    player1 = 0


    # Let op player 0 en 1 / maar WHITE = 1 en BLACK = 2! (zwart begint)
    players = [random_player, alpha_nim_player]
    # players = [alpha_zero.player, alpha_zero.player]

    start = time.time()
    for i in range(1000):
        game = AlphaNim()

        # select player that plays with BLACK (= starting player)
        first_player = random.randint(0, 1)

        while True:
            moves = game.legal_moves()
            if len(moves) == 0:
                # game over
                break

            if first_player == 0:
                if game.current_player == BLACK:
                    move = players[0](game)
                else:
                    move = players[1](game)
            else:
                if game.current_player == BLACK:
                    move = players[1](game)
                else:
                    move = players[0](game)

            game = game.perform_move(move)

        result, _, _ = game.winning_player()
        if result == 0:
            tie += 1
        else:
            if result == BLACK:  # black wins => first player wins
                if first_player == 0:
                    player0 += 1
                else:
                    player1 += 1
            else:
                if first_player == 0:
                    player1 += 1
                else:
                    player0 += 1

    print("P0, P1, TIE", player0, player1, tie)

    end = time.time()
    print("Time consumed in working: ", end - start)