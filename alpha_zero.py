import math
import os

from mcts import MCTS

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
        if os.path.isfile(filename + ".h5") or os.path.isfile(filename + ".index"):
            print("Load model: ", filename)
            self.nn.load_weights(filename)

    def save_nn_values(self):
        filename = MODEL_PATH + self.alpha_game.network_name()
        self.nn.save_weights(filename)

    def predict_best_move(self, full_gamestate):
        # create policy from prediction and legal actions
        legal_actions = self.alpha_game.legal_moves()
        p, v = self.nn.predict(full_gamestate)
        policy = self.alpha_game.p_to_policy(p, legal_actions)

        # find best move in policy
        best_move = None
        best_prior = -math.inf

        for p_action in policy:
            if p_action[0] > best_prior:
                best_prior = p_action[0]
                best_move = p_action[1]

        return best_move # None if there is no move

    # perform training iterations to fill the tree
    def mcts_training(self):
        pass

    def mcts_extract_training_examples(self):
        pass

    def fit(self):
        pass



