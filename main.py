from alpha_reversi import AlphaReversi
from alpha_zero import AlphaZero

if __name__ == '__main__':
    alpha_reversi = AlphaReversi()
    alpha_zero = AlphaZero(alpha_reversi)

    print(alpha_reversi)