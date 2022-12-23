import random
import time

import numpy as np

# Reversi constants
BOARD_SIZE = 8
assert BOARD_SIZE % 2 == 0, "Board size must be an even number"

EMPTY = 0
WHITE = 1
BLACK = 2

DIRECTIONS = [np.array([-1, 0]), np.array([-1, 1]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0]),
                      np.array([1, -1]), np.array([0, -1]), np.array([-1, -1])]


# stateless helper-functions for Reversi
def create_initial_board():
    # empty board
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint32)

    # add initial pieces in middle
    board[BOARD_SIZE//2 - 1, BOARD_SIZE//2 - 1] = WHITE
    board[BOARD_SIZE//2, BOARD_SIZE//2] = WHITE
    board[BOARD_SIZE//2 - 1, BOARD_SIZE//2] = BLACK
    board[BOARD_SIZE//2, BOARD_SIZE//2 - 1] = BLACK

    return board


def on_board(location):
    return 0 <= location[0] < BOARD_SIZE and 0 <= location[1] < BOARD_SIZE


def reversi_to_gamestate_tensor(reversi_game):
    # TODO: implement and move to AI
    print(reversi_game.board & 1)  # white
    print(reversi_game.board >> 1) # black
    print(reversi_game.current_player) # player


class Reversi():
    def __init__(self, board=None, current_player=None):
        if board is None:
            self.board = create_initial_board() # initial game

        else:
            self.board = board
            self.current_player = current_player

        if current_player is None:
            self.current_player = BLACK # black begins

    def get_id(self):
        return tuple(map(tuple, self.board)) + (self.current_player,)

    def find_disks_to_turn(self, new_disk_location):
        location = np.array(new_disk_location)

        disks_to_turn = []

        for direction in DIRECTIONS:
            current_location = location + direction
            current_disks = []
            while on_board(current_location):
                # print(current_location)
                current_location_state = self.board[current_location[0], current_location[1]]
                # stop searching is cell is empty
                if current_location_state == 0:
                    break
                # connected own piece, so everything inbetween (possibly zero) can be turned
                if current_location_state == self.current_player:
                    disks_to_turn += current_disks
                    break

                # add intermediate piece (=from other color)
                current_disks.append(current_location)
                current_location = current_location + direction

        return disks_to_turn

    def legal_moves(self):
        moves = []

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] == 0:
                    # move is legal if it would turn pieces
                    if len(self.find_disks_to_turn((i, j))) > 0:
                        moves.append((i, j))

        return moves

    def perform_move(self, new_disk_location):
        new_board = self.board.copy()

        new_disk_locations = self.find_disks_to_turn(new_disk_location) + [new_disk_location]

        # turn new pieces
        for location in new_disk_locations:
            new_board[location[0], location[1]] = self.current_player

        return Reversi(new_board, self.other_player())

    def other_player(self):
        return BLACK if self.current_player == WHITE else WHITE

    def winning_player(self):
        points_white = np.sum(self.board & 1)
        points_black = np.sum(self.board >> 1)

        if points_white > points_black:
            return WHITE, points_white, points_black

        if points_black > points_white:
            return BLACK, points_white, points_black

        # tie
        return 0, points_white, points_black

    def __str__(self):
        return str(self.board) + "\n" + "PLAYER: " + str(self.current_player)


if __name__ == '__main__':
    game = None

    results = [0, 0, 0]

    start = time.time()

    for i in range(10000):
        game = Reversi()
        while True:
            moves = game.legal_moves()
            if len(moves) == 0:
                # game over
                break

            game = game.perform_move(random.choice(moves))
            #print(game)

        result, _, _ = game.winning_player()
        results[result] += 1

    results.reverse()
    print(results)

    end = time.time()
    print("Time consumed in working: ", end - start)

    # [4199, 5411, 390]

