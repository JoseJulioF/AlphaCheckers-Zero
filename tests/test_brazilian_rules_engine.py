import unittest
import numpy as np

from game import Checkers


class TestBrazilianRulesEngine(unittest.TestCase):
    def setUp(self):
        self.game = Checkers()

    def _empty_board(self):
        return np.zeros((8, 8), dtype=np.int8)

    def test_man_captures_forward_and_backward(self):
        board = self._empty_board()
        board[4, 3] = 1
        board[3, 2] = -1
        board[5, 4] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertIn((((4, 3), (2, 1)),), moves)
        self.assertIn((((4, 3), (6, 5)),), moves)

    def test_mandatory_capture_blocks_simple_moves(self):
        board = self._empty_board()
        board[4, 3] = 1
        board[5, 4] = -1
        board[6, 1] = 1  # teria movimento simples, mas captura é obrigatória

        moves = self.game.get_valid_moves(board, 1)
        self.assertEqual(moves, [(((4, 3), (6, 5)),)])

    def test_multiple_capture_sequence_for_man(self):
        board = self._empty_board()
        board[6, 1] = 1
        board[5, 2] = -1
        board[3, 4] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertIn((((6, 1), (4, 3)), ((4, 3), (2, 5))), moves)

    def test_majority_rule_keeps_only_max_capture_sequences(self):
        board = self._empty_board()
        board[6, 1] = 1
        board[6, 5] = 1
        board[5, 2] = -1
        board[5, 4] = -1
        board[3, 2] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertTrue(all(len(m) == 2 for m in moves))
        self.assertIn((((6, 5), (4, 3)), ((4, 3), (2, 1))), moves)

    def test_flying_king_has_free_long_movement(self):
        board = self._empty_board()
        board[4, 3] = 2

        moves = self.game.get_valid_moves(board, 1)
        self.assertIn(((4, 3), (0, 7)), moves)
        self.assertIn(((4, 3), (7, 0)), moves)

    def test_flying_king_captures_at_distance_with_all_landing_squares(self):
        board = self._empty_board()
        board[5, 0] = 2
        board[3, 2] = -1

        moves = self.game.get_valid_moves(board, 1)
        expected = {
            (((5, 0), (2, 3)),),
            (((5, 0), (1, 4)),),
            (((5, 0), (0, 5)),),
        }
        self.assertTrue(expected.issubset(set(moves)))

    def test_captured_piece_remains_on_board_during_sequence(self):
        board = self._empty_board()
        board[4, 3] = 2
        board[3, 2] = -1
        board[6, 5] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertTrue(moves)
        self.assertTrue(all(len(m) == 1 for m in moves))

    def test_captured_piece_cannot_be_recaptured_in_same_sequence(self):
        board = self._empty_board()
        board[4, 1] = 2
        board[3, 2] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertTrue(moves)
        self.assertTrue(all(len(m) == 1 for m in moves))

    def test_origin_square_is_free_during_sequence(self):
        board = self._empty_board()
        board[4, 3] = 2
        board[3, 2] = -1
        board[5, 0] = -1

        # Sem casa de origem livre, não haveria rota para considerar nova captura
        moves = self.game.get_valid_moves(board, 1)
        self.assertTrue(any(step[0] == (4, 3) for m in moves for step in m))

    def test_promotion_only_at_end_and_new_king_does_not_continue(self):
        board = self._empty_board()
        board[2, 1] = 1
        board[1, 2] = -1
        board[1, 4] = -1

        moves = self.game.get_valid_moves(board, 1)
        self.assertEqual(moves, [(((2, 1), (0, 3)),)])

        result = self.game.apply_move(board, moves[0])
        self.assertEqual(result[0, 3], 2)
        self.assertEqual(result[1, 4], -1)  # não continuou capturando após promoção

    def test_move_representation_is_hashable_and_mcts_compatible(self):
        board = self._empty_board()
        board[6, 1] = 1
        board[5, 2] = -1
        board[3, 4] = -1

        moves = self.game.get_valid_moves(board, 1)
        move_map = {m: idx for idx, m in enumerate(moves)}
        self.assertEqual(len(move_map), len(moves))


if __name__ == "__main__":
    unittest.main()
