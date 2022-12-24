import math

from mcts import MCTS


class AlphaZero():
    def __init__(self, alpha_game):
        self.alpha_game = alpha_game
        self.mcts = MCTS(self.alpha_game)

        self.nn = alpha_game.create_nn()

    # TODO
    # - cleanup policy and normalize (alphazero)
    # - predict and train (alphazero)
    # - simulate games

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



