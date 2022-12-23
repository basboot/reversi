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

# helper class with functionality needed by AlphaZero for playing Reversi
class AlphaReversi():

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

    t_gamestate = tf.keras.backend.constant(np.reshape(dummy_input, (1, 8, 8, 3)))

    p, v = nn.predict(t_gamestate)
    print(p, v)
