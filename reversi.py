import numpy as np

BOARD_SIZE = 8
assert BOARD_SIZE % 2 == 0, "Board size must be an even number"

WHITE = 1
BLACK = 2

import tensorflow as tf

def create_board():
    # empty board
    gamestate = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint32)

    # add initial pieces in middle
    gamestate[BOARD_SIZE//2 - 1, BOARD_SIZE//2 - 1] = WHITE
    gamestate[BOARD_SIZE//2, BOARD_SIZE//2] = WHITE
    gamestate[BOARD_SIZE//2 - 1, BOARD_SIZE//2] = BLACK
    gamestate[BOARD_SIZE//2, BOARD_SIZE//2 - 1] = BLACK

    return gamestate

if __name__ == '__main__':
    gamestate = create_board()
    print((gamestate & 2) >> 1)

    print()


