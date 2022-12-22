import math
import random

from reversi import Reversi

C = 2 # constant to balance exploration vs exploitation (higher = more exploration)

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
            return (self.out_node.v / self.out_node.n) + C * math.sqrt(math.log(self.in_node.n / self.out_node.n))

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

    def play_game(self):
        # move to leaf
        leaf, breadcrumbs = self.moveToLeaf()

        # if already visited, expand leaf first, and move to unvisted child node
        if leaf.n > 0:
            leaf.expand(self.nodes)

            if len(leaf.edges) > 0:
                leaf = leaf.max_ucb_child_node()
                breadcrumbs.append(leaf)
            else:
                # Node is terminal,, so we cannot move further
                pass

        # simulate rest of the game to find a value
        simulation = leaf.game
        while True:
            # TODO: limit depth?
            moves = simulation.legal_moves()
            if len(moves) > 0:
                # TODO: replace with NN
                simulation = simulation.perform_move(random.choice(moves))
            else:
                break
        winning_player, _, _ = simulation.winning_player()

        # backpropagate value
        for breadcrumb in breadcrumbs:
            breadcrumb.n += 1
            value = 0
            if breadcrumb.game.current_player == winning_player:
                value = 1
            if breadcrumb.game.other_player() == winning_player:
                value = -1

            breadcrumb.v += value

    def moveToLeaf(self):
        current_node = self.root

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
    game = Reversi()
    mcts = MCTS(game)
    for i in range(100):
        mcts.play_game()
    mcts.show()

    print(mcts.root.policy())

    # TODO: backtracking controleren