import random
import time

from balanced_player import balanced_player
from greedy_player import greedy_player
from patient_player import patient_player
from random_player import random_player
from reversi import *

if __name__ == '__main__':
    game = None

    tie = 0
    player1 = 0
    player2 = 0

    players = [random_player, balanced_player]

    start = time.time()
    for i in range(1000):
        game = Reversi()
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
            if result == 2: # white wins
                if first_player == 0:
                    player1 += 1
                else:
                    player2 += 1
            else:
                if first_player == 0:
                    player2 += 1
                else:
                    player1 += 1

    print("P1, P2, TIE", player1, player2, tie)

    end = time.time()
    print("Time consumed in working: ", end - start)