import random

import numpy as np

import reversi
from reversi import Reversi
import tensorflow as tf

BOARD_SIZE = reversi.BOARD_SIZE
GAMESTATE_HISTORY_SIZE = 1

# do not change. constants are for readability only
PIECE_TYPE_LAYERS = 2
PLAYER_TURN_LAYER = 1
VALUE_SIZE = 1

# wrapper class with functionality needed by AlphaZero for playing Reversi
class AlphaReversi(Reversi):
    def __init__(self, board=None, current_player=None, full_gamestate=None):
        super().__init__(board, current_player)

        if full_gamestate is None:
            self.full_gamestate = self.init_full_gamestate()
            self.full_gamestate = self.push_move_to_gamestate(self.full_gamestate, self.board, self.current_player)
        else:
            self.full_gamestate = full_gamestate

    # override to also update the full gamestate for alpha
    def perform_move(self, new_disk_location):
        new_board, new_player = self.perform_move_on_board(new_disk_location)
        new_full_gamestate = self.push_move_to_gamestate(self.full_gamestate, new_board, new_player)

        return AlphaReversi(new_board, new_player, new_full_gamestate)

    @staticmethod
    def create_nn():
        # TODO: dummy network, need to select topology

        # use inputs from gamestate_stack
        inputs = tf.keras.layers.Input(
            shape=(BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER))

        t = inputs

        t = tf.keras.layers.Flatten()(t)
        outputPolicy = tf.keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, kernel_initializer='random_normal',
                                  bias_initializer='zeros')(t)
        outputValue = tf.keras.layers.Dense(VALUE_SIZE, kernel_initializer='random_normal',
                                  bias_initializer='zeros')(t)

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
            prior = p[BOARD_SIZE * action[0] + action[1]]
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
            p[BOARD_SIZE * p_action[1][0] + p_action[1][1]] = p_action[0]

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