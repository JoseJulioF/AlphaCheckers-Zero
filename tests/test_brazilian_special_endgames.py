import unittest
import numpy as np

from game import Checkers


class TestBrazilianSpecialEndgames(unittest.TestCase):
    def setUp(self):
        self.game = Checkers()

    def _empty_board(self):
        return np.zeros((8, 8), dtype=np.int8)

    def _play_five_moves_each_side(self, board, white_cycle=((7, 0), (6, 1)), black_cycle=((0, 5), (1, 4))):
        player = 1
        for ply in range(10):
            valid_moves = self.game.get_valid_moves(board, player)
            self.assertTrue(valid_moves, f"Sem jogadas válidas no meio-lance {ply + 1}")

            if player == 1:
                from_sq, to_sq = (white_cycle[0], white_cycle[1]) if board[white_cycle[0]] == 2 else (white_cycle[1], white_cycle[0])
            else:
                from_sq, to_sq = (black_cycle[0], black_cycle[1]) if board[black_cycle[0]] == -2 else (black_cycle[1], black_cycle[0])

            expected_move = (from_sq, to_sq)

            self.assertIn(expected_move, valid_moves, f"Movimento esperado {expected_move} não encontrado.")
            board = self.game.apply_move(board, expected_move)
            player *= -1
            if ply < 9:
                self.assertIsNone(self.game.check_game_over(board, player))

        self.assertEqual(self.game.check_game_over(board, player), 0)

    def test_2_damas_vs_2_damas(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[7, 2] = 2
        board[0, 5] = -2
        board[0, 1] = -2
        self._play_five_moves_each_side(board)

    def test_2_damas_vs_1_dama(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[7, 2] = 2
        board[0, 5] = -2
        self._play_five_moves_each_side(board)

    def test_1_dama_1_homem_vs_1_dama(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[6, 7] = 1
        board[0, 5] = -2
        self._play_five_moves_each_side(board)

    def test_1_dama_vs_1_dama(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[0, 5] = -2
        self._play_five_moves_each_side(board)

    def test_1_dama_vs_1_dama_1_homem(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[0, 5] = -2
        board[1, 0] = -1
        self._play_five_moves_each_side(board)

    def test_3_damas_vs_1_dama_na_grande_diagonal(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[7, 2] = 2
        board[6, 5] = 2
        board[1, 1] = -2
        self._play_five_moves_each_side(board, black_cycle=((1, 1), (0, 0)))

    def test_2_damas_1_homem_vs_1_dama(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[7, 2] = 2
        board[6, 7] = 1
        board[0, 5] = -2
        self._play_five_moves_each_side(board)

    def test_1_dama_2_homens_vs_1_dama(self):
        board = self._empty_board()
        board[7, 0] = 2
        board[6, 3] = 1
        board[6, 7] = 1
        board[0, 5] = -2
        self._play_five_moves_each_side(board)


if __name__ == "__main__":
    unittest.main()
