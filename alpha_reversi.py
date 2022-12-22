import reversi
from reversi import Reversi
import tensorflow as tf

BOARD_SIZE = reversi.BOARD_SIZE
GAMESTATE_HISTORY_SIZE = 1

# do not change. constants are for readability only
PIECE_TYPE_LAYERS = 2
PLAYER_TURN_LAYER = 1
VALUE_SIZE = 1

# wrapper class to add functionality to the game reversi for AlphaZero
class AlphaReversi(Reversi):
    def __init__(self):
        super().__init__()

    def create_nn(self):
        # TODO: dummy network, need to select topology

        # use inputs from gamestate_stack
        inputs = tf.keras.layers.Input(
            shape=(BOARD_SIZE, BOARD_SIZE, PIECE_TYPE_LAYERS * GAMESTATE_HISTORY_SIZE + PLAYER_TURN_LAYER))

        t = inputs

        t = tf.keras.layers.Flatten()(t)
        t = tf.keras.layers.Dense(BOARD_SIZE * BOARD_SIZE + VALUE_SIZE, kernel_initializer='random_normal',
                                  bias_initializer='zeros')(t)
        outputs = t

        model = tf.keras.models.Model(inputs, outputs)

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
    # - convert prediction to policy and value
