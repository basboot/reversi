from mcts import MCTS


class AlphaZero():
    def __init__(self, alpha_game):
        self.alpha_game = alpha_game
        self.mcts = MCTS(self.alpha_game)

    # TODO
    # - cleanup policy and normalize (alphazero)
    # - predict and train (alphazero)
    # - simulate games

    def predict_best_move(self):
        # TODO: add real prediction
        moves = self.alpha_game.legal_moves()
        pass
