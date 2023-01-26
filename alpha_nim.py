import random

import numpy as np
from keras import activations

import nim
from nim import Nim
import tensorflow as tf

BOARD_SIZE = nim.HEAPS
GAMESTATE_HISTORY_SIZE = 1

# do not change. constants are for readability only
PIECE_TYPE_LAYERS = 1
PLAYER_TURN_LAYER = 1
VALUE_SIZE = 1

INPUT_DIMENSION = (BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER)
OUTPUT_DIMENSION = (BOARD_SIZE * BOARD_SIZE,)

# wrapper class with functionality needed by AlphaZero for playing Reversi
class AlphaNim(nim.Nim):
    def __init__(self, board=None, current_player=None, full_gamestate=None):
        super().__init__(board, current_player)

        if full_gamestate is None:
            self.full_gamestate = self.init_full_gamestate()
            self.full_gamestate = self.push_move_to_gamestate(self.full_gamestate, self.board, self.current_player)
        else:
            self.full_gamestate = full_gamestate

        self.input_dimension = INPUT_DIMENSION
        # policy only
        self.output_dimension = OUTPUT_DIMENSION

    def network_name(self):
        return "alpha_nim_test"

    # override to also update the full gamestate for alpha
    def perform_move(self, move):
        new_board, new_player = self.perform_move_on_board(move)
        new_full_gamestate = self.push_move_to_gamestate(self.full_gamestate, new_board, new_player)

        return AlphaNim(new_board, new_player, new_full_gamestate)

    @staticmethod
    def create_nn():
        # TODO: dummy network, need to select topology

        # use inputs from gamestate_stack
        inputs = tf.keras.layers.Input(
            shape=INPUT_DIMENSION)

        # input
        t = inputs
        t = tf.keras.layers.Flatten()(t)

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)

        # hidden
        t = tf.keras.layers.Dense(16, kernel_initializer=initializer,
                                             bias_initializer='zeros', activation=activations.relu)(t)
        t = tf.keras.layers.Dense(32, kernel_initializer=initializer,
                                  bias_initializer='zeros', activation=activations.relu)(t)

        
        # output
        outputPolicy = tf.keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, kernel_initializer=initializer,
                                  bias_initializer='zeros', activation=activations.sigmoid)(t)
        outputValue = tf.keras.layers.Dense(VALUE_SIZE, kernel_initializer=initializer,
                                  bias_initializer='zeros', activation=activations.tanh)(t)

        model = tf.keras.models.Model(inputs, [outputPolicy, outputValue])

        model.compile(
            optimizer='adam',
            # loss=tf.keras.losses.Huber(), # SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
        )
        model.summary()

        return model

    # filter legal actions from prediction and normalize
    @staticmethod
    def p_to_policy(p, legal_actions):
        policy = []
        sum_priors = 0
        for action in legal_actions:
            prior = p[BOARD_SIZE * action[0] + action[1] - 1]
            sum_priors += prior
            policy.append([prior, action])

        # normalization only possible when sum of priors is not equal to zero
        if sum_priors > 0:
            for p_action in policy:
                p_action[0] /= sum_priors

        return policy

    @staticmethod
    def policy_to_p(policy):
        p = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for p_action in policy:
            p[BOARD_SIZE * p_action[1][0] + p_action[1][1] - 1] = p_action[0]

        return p

    @staticmethod
    def init_full_gamestate():
        return np.zeros((BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER))

    @staticmethod
    def push_move_to_gamestate(gamestate_stack, board, current_player):
        # do not modify original
        gamestate_stack = np.copy(gamestate_stack)

        if GAMESTATE_HISTORY_SIZE > 1:
            gamestate_stack[:, :, :-(PIECE_TYPE_LAYERS + PLAYER_TURN_LAYER)] = \
                gamestate_stack[:, :, PIECE_TYPE_LAYERS:-PLAYER_TURN_LAYER]

        gamestate_stack[:, :, -PLAYER_TURN_LAYER] = \
            np.zeros((BOARD_SIZE, BOARD_SIZE)) if current_player == nim.BLACK else np.ones((BOARD_SIZE, BOARD_SIZE))

        np_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for i in range(len(board)): # convert list of heap, number to matrix
            if board[i] == 0:
                continue
            np_board[i, board[i] - 1] = 1

        gamestate_stack[:, :, -(PLAYER_TURN_LAYER + 1)] = np_board

        return gamestate_stack

    # TODO:
    # - gamestate to tensor
    # - push gamestate in tensor history stack
    # - policy and value to tensor
    # - convert prediction to policy and value
    # - filter legal actions / renormalize policy (in alphazero?)


if __name__ == '__main__':
    nn = AlphaNim.create_nn()

    dummy_input = np.zeros(INPUT_DIMENSION)

    print(dummy_input.shape)

    t_gamestate = tf.keras.backend.constant(np.reshape(dummy_input, (1,) + INPUT_DIMENSION))

    p, v = nn.predict(t_gamestate)
    print(p, v)

    game = nim.Nim()
    print(AlphaNim.p_to_policy(p[0], game.legal_moves()))

    policy = [(0.021586762545612, (0, 1)), (0.007797914222943548, (1, 1)), (0.010900880090319006, (1, 2)), (0.008137910482984687, (2, 1)), (0.010889880211317676, (2, 2)), (0.010867880453315014, (3, 1)), (0.006282930887760235, (3, 2)), (0.9124209633694029, (3, 3)), (0.0111148777363449, (3, 4))]


    print(AlphaNim.policy_to_p(policy))

    full_game = AlphaNim.init_full_gamestate()
    full_game = AlphaNim.push_move_to_gamestate(full_game, game.board, game.current_player)
    layer = -2
    print(full_game[:, :, layer])
    game = game.perform_move(random.choice(game.legal_moves()))
    full_game = AlphaNim.push_move_to_gamestate(full_game, game.board, game.current_player)

    layer = -3
    print(full_game[:, :, layer])