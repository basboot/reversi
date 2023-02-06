def human_player(game):
    print(game)

    moves = game.legal_moves()
    for move in moves:
        print(move)
    

    print("Welke stapel? (0-3)")
    heap = int(input())

    print("Hoeveel? (0-%d)" % game.board[heap])
    number = int(input())

    return heap, number