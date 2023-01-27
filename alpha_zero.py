import math
import os
import random

from mcts import MCTS
from alpha_nim import *

MODEL_PATH = "./models/"

N_SAMPLES = 100000
N_VALIDATION = 1000
N_SIMULATIONS = 1000000
N_GAMES = 100

class AlphaZero():
    def __init__(self, alpha_game, load_nn=True):
        self.alpha_game = alpha_game
        self.mcts = MCTS(self.alpha_game)

        # learning network, update
        self.nn = alpha_game.create_nn()
        # copy, to keep current best
        # TODO: rename/refactor when ready (other network should be named target, or maybe even another name)
        self.nn_target = alpha_game.create_nn()

        if load_nn:
            self.load_nn_values()

        # copy network
        self.nn_target.set_weights(self.nn.get_weights())

    # TODO
    # - cleanup policy and normalize (alphazero)
    # - predict and train (alphazero)
    # - simulate games

    def load_nn_values(self):
        # load model weights from disk, if exists
        filename = MODEL_PATH + self.alpha_game.network_name
        print("LOAD: ", filename)
        # sometimes keras replaces .h5 with multiple data files and a .index
        # TODO: check if extension .tf is correct
        if os.path.isfile(filename + ".tf") or os.path.isfile(filename + ".h5") or os.path.isfile(filename + ".index"):
            print("Load model: ", filename)
            self.nn.load_weights(filename)

    def save_nn_values(self):
        filename = MODEL_PATH + self.alpha_game.network_name
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

    def predict_best_move(self, game, nn=None):
        if nn is None:
            nn = self.nn

        # create policy from prediction and legal actions
        legal_actions = game.legal_moves()
        # p, _ = self.nn.predict(tf.reshape(game.full_gamestate, (1,) + self.alpha_game.input_dimension), verbose=0)
        p, v = nn(tf.reshape(game.full_gamestate, (1,) + self.alpha_game.input_dimension), training=False)

        print("p = ", p)
        print("v = ", v)
        # one sample, one prediction
        policy = self.alpha_game.p_to_policy(p[0], legal_actions)

        print("p_clean = ", policy)

        return self.best_move_from_policy(policy)

    def predict_v(self, game):
        #_, v = self.nn.predict(tf.reshape(game.full_gamestate, (1,) + self.alpha_game.input_dimension), verbose=0)

        # Use model without predict for faster prediction on small datasets
        # https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods
        _, v = self.nn(tf.reshape(game.full_gamestate, (1,) + self.alpha_game.input_dimension), training=False)

        return v

    # perform training iterations to fill the tree
    def mcts_training(self, game=None, n_simulations=None):
        if game is None:
            game = self.alpha_game

        if game.get_id() not in self.mcts.nodes:
            print("Gamestate not in MCTS, create new MCTS with this gamestate as root")
            self.mcts = MCTS(game)


        for n in range(n_simulations):
            self.mcts.simulate_game(self.mcts.nodes[game.get_id()])

    def mcts_extract_training_examples(self, n_samples=N_SAMPLES, n_validation=N_VALIDATION):

        # arrays with numpy objects in correct shape
        x, y_p, y_v = self.mcts.create_training_samples(n_samples)
        v_x, v_y_p, v_y_v = self.mcts.create_training_samples(n_validation)

        # print(len(x))

        tf_x = tf.stack(x)
        tf_y_p = tf.stack(y_p)
        tf_y_v = tf.stack(y_v)

        tf_v_x = tf.stack(v_x)
        tf_v_y_p = tf.stack(v_y_p)
        tf_v_y_v = tf.stack(v_y_v)

        return tf_x, (tf_y_p, tf_y_v), tf_v_x, (tf_v_y_p, tf_v_y_v)

    def nn_fit_training_examples(self, n_samples=N_SAMPLES, n_validation=N_VALIDATION):
        x_train, y_train, x_validation, y_validation = self.mcts_extract_training_examples(n_samples, n_validation)
        self.nn.fit(x_train, y_train, epochs=10, validation_data = (x_validation,y_validation))

    def nn_compete(self, n_games=N_GAMES):
        # compete normal and target n games, winner survives

        players = [self.nn, self.nn_target]
        points = [0, 0]

        for i in range(n_games):
            game = AlphaNim()
            # select player that plays with BLACK (= starting player)
            first_player = random.randint(0, 1)


            while True:
                moves = game.legal_moves()
                if len(moves) == 0:
                    # game over
                    break

                current_player = (game.current_player + first_player) % 2
                move = self.predict_best_move(game, players[current_player])
                game = game.perform_move(move)

            winning_player, _, _ = game.winning_player()
            # don't need to count ties

            if winning_player > 0:
                points[(winning_player + first_player) % 2] += 1

        # copy winner
        print("RESULT OF COMPETITION: ", points[0], "-", points[1])
        if points[0] > points[1]:
            self.nn_target.set_weights(self.nn.get_weights())
        else:
            self.nn.set_weights(self.nn_target.get_weights())


    def player(self, game, prediction_only=False, mcts_only=False, rollout=True, always_renew_mcts=True, nn_compete=False):

        if always_renew_mcts:
            self.mcts = MCTS(game)

        # prediction only = play without learning
        if not prediction_only:
            # build mtcs
            # if game not in mcts, replace mcts
            if game.get_id() not in self.mcts.nodes:
                print("Gamestate not in MCTS, create new MCTS with this gamestate as root")
                self.mcts = MCTS(game)

            assert game.get_id() in self.mcts.nodes, "node not in mcts"

            N_SIMULATIONS = 1000
            for n in range(N_SIMULATIONS):
                self.mcts.simulate_game(self.mcts.nodes[game.get_id()],
                                        predict_v_function=None if rollout else self.predict_v)

            has_policy = self.mcts.nodes[game.get_id()].has_policy()

            # mcts only: use montecarlo tree to select best move, instead of nn
            # TODO: maybe mtc can also be used for learning
            if mcts_only:
                if has_policy:
                    return self.best_move_from_policy(self.mcts.nodes[game.get_id()].policy())
                else:
                    print("no policy, return random move")
                    return random.choice(game.legal_moves())

            # train nn from expanded tree
            self.nn_fit_training_examples(n_samples=1000, n_validation=100)

        # compete nn after update to choose between new and old version
        if nn_compete:
            self.nn_compete()

        return self.predict_best_move(game)

if __name__ == '__main__':
    game = AlphaNim([1, 2, 3, 4])
    alpha_zero = AlphaZero(game, load_nn=False)

    alpha_zero.load_nn_values()

    alpha_zero.mcts_training(game=None, n_simulations=10000)

    alpha_zero.nn_fit_training_examples(n_samples=10000, n_validation=100)

    alpha_zero.save_nn_values()

    # move = alpha_zero.player(game)

    alpha_zero.predict_best_move(game)

    print("p_mcts", alpha_zero.mcts.root.policy())
    print("v_mcts", alpha_zero.mcts.root.v / alpha_zero.mcts.root.n)

    #alpha_zero.mcts.show()

    # print(move)