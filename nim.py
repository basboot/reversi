import random
import time

import numpy as np

# Reversi constants
from mcts import MCTS

HEAPS = 4

EMPTY = 0
WHITE = 1
BLACK = 2


# stateless helper-functions for Reversi
def create_initial_board():
    # empty board
    board = list(range(1, HEAPS + 1))

    return board


class Nim():
    def __init__(self, board=None, current_player=None):
        if board is None:
            self.board = create_initial_board() # initial game

        else:
            self.board = board
            self.current_player = current_player

        if current_player is None:
            self.current_player = BLACK # black begins

    def get_id(self):
        return tuple(self.board) + (self.current_player,)

    def legal_moves(self):
        moves = []

        for i in range(HEAPS):
            for j in range(self.board[i]):
                moves.append((i, j + 1))

        return moves

    def perform_move_on_board(self, move):
        new_board = self.board.copy()

        new_board[move[0]] -= move[1]

        return new_board, self.other_player()

    def perform_move(self, move):
        new_board, new_player = self.perform_move_on_board(move)

        return Nim(new_board, new_player)

    def other_player(self):
        return BLACK if self.current_player == WHITE else WHITE

    def winning_player(self):
        # not completely safe
        assert len(self.legal_moves()) == 0, "No winning player yet"
        return self.other_player(), 0, 0

    def __str__(self):
        return str("PLAYER: " + str(self.current_player) + " " + str(self.board))


if __name__ == '__main__':
    game = None

    results = [0, 0, 0]

    start = time.time()

    for i in range(0):
        game = Nim()
        print("Start New Game")
        print(game)
        while True:
            moves = game.legal_moves()
            if len(moves) == 0:
                # game over
                break

            move = random.choice(moves)
            print("Player takes", move[1], "from heap ", (move[0] + 1))
            game = game.perform_move(move)
            print(game)

        result, _, _ = game.winning_player()
        results[result] += 1
        print("Win for player:", result)

    results.reverse()
    print(results)

    end = time.time()
    print("Time consumed in working: ", end - start)

    # [4199, 5411, 390]

    game = Nim([1, 2, 2, 0], BLACK) #Nim([1, 2, 2, 4], BLACK)

    mcts = MCTS(game)
    N_SIMULATIONS = 10000
    for n in range(N_SIMULATIONS):
        mcts.simulate_game(mcts.nodes[game.get_id()])

    mcts.show()

    policy = mcts.nodes[game.get_id()].policy()

    print(policy)

    print("MCTS simulation")
    print(game)
    p_total = 0
    for move in policy:
        p_total += move[0]
        print("p_win = ",move[0],": take", move[1][1], "from heap", (move[1][0]+1))

    print(p_total)

