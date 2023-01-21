import math
import os
import random

from mcts import MCTS
from alpha_nim import *

MODEL_PATH = "./models/"


class AlphaZero():
    def __init__(self, alpha_game):
        self.alpha_game = alpha_game
        self.mcts = MCTS(self.alpha_game)

        self.nn = alpha_game.create_nn()

    # TODO
    # - cleanup policy and normalize (alphazero)
    # - predict and train (alphazero)
    # - simulate games

    def load_nn_values(self):
        # load model weights from disk, if exists
        filename = MODEL_PATH + self.alpha_game.network_name()
        print("LOAD: ", filename)
        # sometimes keras replaces .h5 with multiple data files and a .index
        # TODO: check if extension .tf is correct
        if os.path.isfile(filename + ".tf") or os.path.isfile(filename + ".h5") or os.path.isfile(filename + ".index"):
            print("Load model: ", filename)
            self.nn.load_weights(filename)

    def save_nn_values(self):
        filename = MODEL_PATH + self.alpha_game.network_name()
        # Use tf format to also save optimizer state
        #  https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training
        self.nn.save_weights(filename, save_format='tf')

    @staticmethod
    def best_move_from_policy(policy):
        # find best move in policy
        best_move = None
        best_prior = -math.inf

        for p_action in policy:
            if p_action[0] > best_prior:
                best_prior = p_action[0]
                best_move = p_action[1]
        return best_move  # None if there is no move

    def predict_best_move(self, game):
        # create policy from prediction and legal actions
        legal_actions = game.legal_moves()
        p, v = self.nn.predict(tf.reshape(game.full_gamestate, (1,) + self.alpha_game.input_dimension), verbose=0)

        # one sample, one prediction
        policy = self.alpha_game.p_to_policy(p[0], legal_actions)

        return self.best_move_from_policy(policy)

    # perform training iterations to fill the tree
    def mcts_training(self, game=None):
        if game is None:
            game = self.alpha_game

        if game.get_id() not in self.mcts.nodes:
            print("Gamestate not in MCTS, create new MCTS with this gamestate as root")
            self.mcts = MCTS(game)

        N_SIMULATIONS = 100000
        for n in range(N_SIMULATIONS):
            self.mcts.simulate_game(self.mcts.nodes[game.get_id()])

    def mcts_extract_training_examples(self):
        N_SAMPLES = 10000
        N_VALIDATION = 100

        # arrays with numpy objects in correct shape
        x, y_p, y_v = self.mcts.create_training_samples(N_SAMPLES)
        v_x, v_y_p, v_y_v = self.mcts.create_training_samples(N_VALIDATION)

        # print(len(x))

        tf_x = tf.stack(x)
        tf_y_p = tf.stack(y_p)
        tf_y_v = tf.stack(y_v)

        tf_v_x = tf.stack(v_x)
        tf_v_y_p = tf.stack(v_y_p)
        tf_v_y_v = tf.stack(v_y_v)

        return tf_x, (tf_y_p, tf_y_v), tf_v_x, (tf_v_y_p, tf_v_y_v)

    def nn_fit_training_examples(self):
        x_train, y_train, x_validation, y_validation = self.mcts_extract_training_examples()
        self.nn.fit(x_train, y_train, epochs=10, validation_data = (x_validation,y_validation))

    def player(self, game):
        # if game not in mcts, replace mcts
        if game.get_id() not in self.mcts.nodes:
            print("Gamestate not in MCTS, create new MCTS with this gamestate as root")
            self.mcts = MCTS(game)

        assert game.get_id() in self.mcts.nodes, "node not in mcts"

        N_SIMULATIONS = 10
        for n in range(N_SIMULATIONS):
            self.mcts.simulate_game(self.mcts.nodes[game.get_id()])

        has_policy = self.mcts.nodes[game.get_id()].has_policy()

        if has_policy:
            return self.best_move_from_policy(self.mcts.nodes[game.get_id()].policy())
        else:
            print("no policy, return random move")
            return random.choice(game.legal_moves())

if __name__ == '__main__':
    game = AlphaNim()
    alpha_zero = AlphaZero(game)

    alpha_zero.load_nn_values()

    alpha_zero.mcts_training()

    alpha_zero.nn_fit_training_examples()

    alpha_zero.save_nn_values()

    move = alpha_zero.player(game)

    print(alpha_zero.predict_best_move())

    print(move)