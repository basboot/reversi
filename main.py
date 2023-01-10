import random
import time

from greedy_player import greedy_player
from patient_player import patient_player
from random_player import random_player
from reversi import *

from alpha_zero import *
from alpha_reversi import *

if __name__ == '__main__':
    game = None

    tie = 0
    player0 = 0
    player1 = 0

    alpha_zero = AlphaZero(AlphaReversi())

    # Let op player 0 en 1 / maar WHITE = 1 en BLACK = 2! (zwart begint)
    players = [patient_player, random_player]
    # players = [random_player, random_player]


    start = time.time()
    for i in range(100):
        game = AlphaReversi()

        # select player that plays with BLACK
        first_player = 0#random.randint(0, 1)

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
            if result == BLACK: # black wins == first player wins
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