import random

import numpy as np
from keras import activations

import reversi
from nn_helper_functions import relu_bn, residual_block
from reversi import Reversi
import tensorflow as tf

BOARD_SIZE = reversi.BOARD_SIZE
GAMESTATE_HISTORY_SIZE = 1

# do not change. constants are for readability only
PIECE_TYPE_LAYERS = 2
PLAYER_TURN_LAYER = 1
VALUE_SIZE = 1

INPUT_DIMENSION = (BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER)
OUTPUT_DIMENSION = (BOARD_SIZE, BOARD_SIZE)

# wrapper class with functionality needed by AlphaZero for playing Reversi
class AlphaReversi(Reversi):
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

        self.network_name = "alpha_reversi_test"

    def set_network_name(self, name):
        self.network_name = name

    # override to also update the full gamestate for alpha
    def perform_move(self, new_disk_location):
        new_board, new_player = self.perform_move_on_board(new_disk_location)
        new_full_gamestate = self.push_move_to_gamestate(self.full_gamestate, new_board, new_player)

        return AlphaReversi(new_board, new_player, new_full_gamestate)

    def new_game(self):
        return AlphaReversi()

    @staticmethod
    def create_nn():
        # TODO: dummy network, need to select topology

        # use inputs from gamestate_stack
        inputs = tf.keras.layers.Input(
            shape=INPUT_DIMENSION)

        t = inputs


        BLOCKS = [1]
        FILTERS = [128]

        # conv filter (pre-filter to connect to resnet)

        t = tf.keras.layers.BatchNormalization()(t)

        t = tf.keras.layers.Conv2D(kernel_size=3,
                                   strides=1,
                                   filters=FILTERS[0],
                                   padding="same")(t)
        t = relu_bn(t)

        # residual network

        num_blocks_list = BLOCKS
        num_filters_list = FILTERS
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                # downsampling needed to match in and output channels of each resnet
                t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters_list[i])

        # output

        # value
        # TODO: split tail of network

        t = tf.keras.layers.BatchNormalization()(t)

        t_pol = tf.keras.layers.Conv2D(kernel_size=1,
                                   strides=1,
                                   filters=32,

                                   padding="same")(t)

        t_pol = tf.keras.layers.Conv2D(kernel_size=1,
                                       strides=1,
                                       filters=1,
                                       activation=activations.sigmoid,
                                       padding="same")(t_pol)

        t_val = tf.keras.layers.Conv2D(kernel_size=1,
                                       strides=1,
                                       filters=3,
                                       activation=activations.relu,
                                       padding="same")(t)


        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)

        # TODO: do we need to flatten?
        # TODO: different paths for p and v: https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a
        t_val = tf.keras.layers.Flatten()(t_val)

        # output
        outputPolicy = t_pol
        outputValue = tf.keras.layers.Dense(VALUE_SIZE, kernel_initializer=initializer,
                                  bias_initializer='zeros', activation=activations.tanh)(t_val)


        model = tf.keras.models.Model(inputs, [outputPolicy, outputValue])

        model.compile(
            optimizer='adam',
            # loss=tf.keras.losses.Huber(), # SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.MeanSquaredError()
        )
        model.summary()

        return model

    # filter legal actions from prediction and normalize
    @staticmethod
    def p_to_policy(p, legal_actions):
        policy = []
        sum_priors = 0
        for action in legal_actions:
            prior = p[action[0], action[1]]
            sum_priors += prior
            policy.append([prior, action])

        # normalization only possible when sum of priors is not equal to zero
        if sum_priors > 0:
            for p_action in policy:
                p_action[0] /= sum_priors

        return policy

    @staticmethod
    def policy_to_p(policy):
        p = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for p_action in policy:
            p[p_action[1][0], p_action[1][1]] = p_action[0]

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
            np.zeros((BOARD_SIZE, BOARD_SIZE)) if current_player == reversi.BLACK else np.ones((BOARD_SIZE, BOARD_SIZE))
        gamestate_stack[:, :, -(PLAYER_TURN_LAYER + 1)] = (board & 1) # white pieces
        gamestate_stack[:, :, -(PLAYER_TURN_LAYER + 2)] = (board >> 1) # black pieces

        return gamestate_stack

    # TODO:
    # - gamestate to tensor
    # - push gamestate in tensor history stack
    # - policy and value to tensor
    # - convert prediction to policy and value
    # - filter legal actions / renormalize policy (in alphazero?)


if __name__ == '__main__':
    nn = AlphaReversi.create_nn()

    dummy_input = np.zeros((BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER))

    print(dummy_input.shape)

    t_gamestate = tf.keras.backend.constant(np.reshape(dummy_input, (1, BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER)))

    p, v = nn.predict(t_gamestate)
    print(p, v)

    game = Reversi()
    print(AlphaReversi.p_to_policy(p[0], game.legal_moves()))

    policy = [(0.29292929292929293, (2, 3)), (0.26262626262626265, (3, 2)), (0.23232323232323232, (4, 5)),
              (0.21212121212121213, (5, 4))]

    print(AlphaReversi.policy_to_p(policy))

    full_game = AlphaReversi.init_full_gamestate()
    full_game = AlphaReversi.push_move_to_gamestate(full_game, game.board, game.current_player)
    game = game.perform_move(random.choice(game.legal_moves()))
    full_game = AlphaReversi.push_move_to_gamestate(full_game, game.board, game.current_player)

    layer = -3
    print(full_game[:, :, layer])