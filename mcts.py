import math
import random

import numpy as np

from alpha_reversi import AlphaReversi
from reversi import Reversi

C = 1 # constant to balance exploration vs exploitation (higher = more exploration)

class Edge():
    def __init__(self, action, in_node, out_node):
        self.action = action
        self.in_node = in_node
        self.out_node = out_node

    def ucb(self):
        # unvisited nodes have ucb infinity (because of dividing by n in the formula)
        if self.out_node.n == 0:
            return math.inf
        else:
            # exploitation average value of child node
            # exoploitation constant x square root of natural log of parent visits divided by child visits
            # print(self.in_node.n, self.out_node.n)
            visit_ratio = self.in_node.n / self.out_node.n
            # TODO: the graph is not acyclic because we avoid duplicate nodes... should this be changed?
            if visit_ratio < 1:
                visit_ratio = 1

            return (self.out_node.v / self.out_node.n) + C * math.sqrt(math.log(visit_ratio))

class Node():
    def __init__(self, game):
        self.game = game
        self.v = 0
        self.n = 0
        self.edges = []

    def get_id(self):
        return self.game.get_id()

    def has_policy(self):
        # node only has a policy if it has been expanded and all edges have been visited
        if len(self.edges) == 0:
            return False
        for edge in self.edges:
            if edge.out_node.n == 0:
                return False
        return True

    def policy(self):
        assert len(self.edges) > 0, "Cannot calculate policy. Node not expanded"

        # sum up edge visits
        total_n = 0
        for edge in self.edges:
            assert edge.out_node.n > 0, "Cannot calculate policy. Not all edges visited."
            total_n += edge.out_node.n

        policy = []
        for edge in self.edges:
            # literature mentions steering on n is good enough
            p = edge.out_node.n / total_n

            policy.append((p, edge.action))

        return policy

    def max_ucb_child_node(self):
        assert len(self.edges) > 0, "Cannot calculate best ucb. Node not expanded"

        max_ucb = -math.inf
        max_child = None

        for edge in self.edges:
            ucb = edge.ucb()
            if ucb > max_ucb:
                max_ucb = ucb
                max_child = edge.out_node

        return max_child

    def expand(self, nodes):
        for move in self.game.legal_moves():
            next_game = self.game.perform_move(move)
            next_node = Node(next_game)
            # if node already exists, connect node to existing node instead of new node
            if next_node.get_id() in nodes:
                next_node = nodes[next_node.get_id()]
            else:
                nodes[next_node.get_id()] = next_node

            next_edge = Edge(move, self, next_node)
            self.edges.append(next_edge)

    def __str__(self):
        return "value: %d, n_visits: %d, n_edges %d" % (self.v, self.n, len(self.edges))


class MCTS():
    def __init__(self, game):
        self.root = Node(game)
        self.nodes = {self.root.get_id(): self.root}

    def create_training_samples(self, n):

        nodes_list = list(self.nodes.keys())

        x = np.zeros((n,) + self.root.game.input_dimension)
        y_p = np.zeros((n,) + self.root.game.output_dimension)
        y_v = np.zeros(n)

        i = 0
        while i < n:
            random_node = self.nodes[random.choice(nodes_list)]
            if not random_node.has_policy():
                continue

            gamestate = random_node.game.full_gamestate
            policy = random_node.policy()
            v = random_node.v / random_node.n

            x[i] = gamestate
            y_p[i] = self.root.game.policy_to_p(policy)
            y_v[i] = v

            i+=1

        return x, y_p, y_v


    def rollout(self, game):
        # TODO: use nn to predict the value
        while True:
            # TODO: limit depth?
            moves = game.legal_moves()
            if len(moves) > 0:
                game = game.perform_move(random.choice(moves))
            else:
                break

        winning_player, _, _ = game.winning_player()

        value = 0
        if self.root.game.current_player == winning_player:
            value = 1
        if self.root.game.other_player() == winning_player:
            value = -1

        return value

    # if predict_v_function is specified it will be used, else a random rollout will be performed to find v
    def simulate_game(self, current_node=None, predict_v_function=None):
        if current_node is None:
            current_node = self.root

        assert current_node.get_id() in self.nodes, "Cannot simulate starting from unknown node"

        # move to leaf
        leaf, breadcrumbs = self.moveToLeaf(current_node)

        # if already visited, expand leaf first, and move to unvisted child node
        if leaf.n > 0:
            # TODO: use nn to fill leaves with policy (find out when this is used!) and use v
            leaf.expand(self.nodes)

            if len(leaf.edges) > 0:
                leaf = leaf.max_ucb_child_node()
                breadcrumbs.append(leaf)
            else:
                # Node is terminal,, so we cannot move further
                pass

        # rollout rest of the game to find a value
        game = leaf.game

        # use prediction function if one is specified, or do a random rollout
        # not in paper: when gamestate is terminal also do a 'rollout' to help
        # the nn learn the winning and losing states
        if predict_v_function is None or len(game.legal_moves()) == 0:
            value = self.rollout(game) # value for current player
        else:
            value = predict_v_function(game)
            # TODO: sign must be adjusted for root player (see below) to match update
            if game.current_player == self.root.game.current_player:
                value *= -1

        # backpropagate value
        for breadcrumb in breadcrumbs:
            breadcrumb.n += 1

            # same player gets same reward, other gets negated reward
            # TODO: I reversed this because ucb is calculated based on the next state
            if breadcrumb.game.current_player == self.root.game.current_player:
                breadcrumb.v -= value
            else:
                breadcrumb.v += value

    def moveToLeaf(self, current_node):
        breadcrumbs = [current_node]

        while len(current_node.edges) > 0:
            current_node = current_node.max_ucb_child_node()
            breadcrumbs.append(current_node)

        return current_node, breadcrumbs

    def show(self):
        print(self.root)
        self.show_children(self.root, 1)

    def show_children(self, node, level):
        for edge in node.edges:
            print(level, edge.out_node)
            self.show_children(edge.out_node, level + 1)

if __name__ == '__main__':
    game = AlphaReversi()
    mcts = MCTS(game)
    for i in range(100):
        mcts.simulate_game()
    #mcts.show()

    print(mcts.root.policy())

    x, y_p, y_v = mcts.create_training_samples(2)
    print(y_p)

    # TODO: backtracking controleren