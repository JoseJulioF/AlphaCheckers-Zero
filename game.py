import numpy as np

BOARD_SIZE = 8
SPECIAL_ENDGAME_HALF_MOVES_LIMIT = 10
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class Checkers:
    def __init__(self):
        self._special_endgame_key = None
        self._special_endgame_halfmoves = 0
        self._special_endgame_draw = False

    def get_initial_board(self):
        self.reset_special_endgame_state()
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r, c] = -1
        for r in range(5, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r, c] = 1
        return board

    def _is_inside(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _is_promotion_row(self, r, player):
        return (player == 1 and r == 0) or (player == -1 and r == BOARD_SIZE - 1)

    def _dynamic_cell(self, board, r, c, current_pos, vacated):
        if (r, c) == current_pos:
            return board[current_pos]
        if (r, c) in vacated:
            return 0
        return board[r, c]

    def _get_simple_moves(self, board, r, c):
        moves = []
        piece = int(board[r, c])
        player = int(np.sign(piece))

        if abs(piece) == 1:
            directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if self._is_inside(nr, nc) and board[nr, nc] == 0:
                    moves.append(((r, c), (nr, nc)))
            return moves

        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            while self._is_inside(nr, nc) and board[nr, nc] == 0:
                moves.append(((r, c), (nr, nc)))
                nr += dr
                nc += dc
        return moves

    def _get_man_captures(self, board, current_pos, player, captured, vacated):
        r, c = current_pos
        captures = []
        for dr, dc in DIAGONALS:
            mid_r, mid_c = r + dr, c + dc
            end_r, end_c = r + 2 * dr, c + 2 * dc
            if not self._is_inside(mid_r, mid_c) or not self._is_inside(end_r, end_c):
                continue
            mid_val = self._dynamic_cell(board, mid_r, mid_c, current_pos, vacated)
            end_val = self._dynamic_cell(board, end_r, end_c, current_pos, vacated)
            if mid_val * player < 0 and (mid_r, mid_c) not in captured and end_val == 0:
                captures.append(((end_r, end_c), (mid_r, mid_c)))
        return captures

    def _get_king_captures(self, board, current_pos, player, captured, vacated):
        r, c = current_pos
        captures = []
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            while self._is_inside(nr, nc) and self._dynamic_cell(board, nr, nc, current_pos, vacated) == 0:
                nr += dr
                nc += dc
            if not self._is_inside(nr, nc):
                continue

            enemy_cell = self._dynamic_cell(board, nr, nc, current_pos, vacated)
            if enemy_cell * player >= 0:
                continue
            if (nr, nc) in captured:
                continue

            enemy_pos = (nr, nc)
            lr, lc = nr + dr, nc + dc
            while self._is_inside(lr, lc):
                landing_cell = self._dynamic_cell(board, lr, lc, current_pos, vacated)
                if landing_cell != 0:
                    break
                captures.append(((lr, lc), enemy_pos))
                lr += dr
                lc += dc
        return captures

    def _find_jump_sequences(self, board, current_pos, piece, captured=frozenset(), vacated=frozenset(), path=()):
        player = int(np.sign(piece))
        options = self._get_man_captures(board, current_pos, player, captured, vacated) \
            if abs(piece) == 1 else \
            self._get_king_captures(board, current_pos, player, captured, vacated)

        if not options:
            return [path] if path else []

        all_paths = []
        for landing_pos, captured_pos in options:
            step = (current_pos, landing_pos)
            next_path = path + (step,)
            next_captured = captured | {captured_pos}
            next_vacated = vacated | {current_pos}

            if abs(piece) == 1 and self._is_promotion_row(landing_pos[0], player):
                all_paths.append(next_path)
                continue

            continuations = self._find_jump_sequences(
                board,
                landing_pos,
                piece,
                captured=next_captured,
                vacated=next_vacated,
                path=next_path
            )
            if continuations:
                all_paths.extend(continuations)
            else:
                all_paths.append(next_path)
        return all_paths

    def _get_all_jumps(self, board, player):
        all_jumps = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] * player > 0:
                    piece = int(board[r, c])
                    jumps = self._find_jump_sequences(board, (r, c), piece)
                    if jumps:
                        all_jumps.extend(jumps)
        if not all_jumps:
            return []
        max_len = max(len(j) for j in all_jumps)
        return [j for j in all_jumps if len(j) == max_len]

    def get_valid_moves(self, board, player):
        jumps = self._get_all_jumps(board, player)
        if jumps:
            return jumps
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] * player > 0:
                    moves.extend(self._get_simple_moves(board, r, c))
        return moves

    def apply_move(self, board, move, track_special_endgame=True):
        b_ = np.copy(board)
        is_sequence = isinstance(move, (list, tuple)) and len(move) > 0 and \
            isinstance(move[0], tuple) and len(move[0]) == 2 and isinstance(move[0][0], tuple)
        sub_moves = tuple(move) if is_sequence else (move,)
        captured_positions = []

        for (r1, c1), (r2, c2) in sub_moves:
            piece = int(b_[r1, c1])
            if piece == 0:
                continue

            player = int(np.sign(piece))
            dr = 1 if r2 > r1 else -1
            dc = 1 if c2 > c1 else -1

            b_[r2, c2] = piece
            b_[r1, c1] = 0

            rr, cc = r1 + dr, c1 + dc
            while (rr, cc) != (r2, c2):
                if b_[rr, cc] * player < 0 and (rr, cc) not in captured_positions:
                    captured_positions.append((rr, cc))
                    break
                rr += dr
                cc += dc

        for rr, cc in captured_positions:
            b_[rr, cc] = 0

        r_final, c_final = sub_moves[-1][1]
        p_final = int(b_[r_final, c_final])
        if p_final == 1 and r_final == 0:
            b_[r_final, c_final] = 2
        if p_final == -1 and r_final == BOARD_SIZE - 1:
            b_[r_final, c_final] = -2

        if track_special_endgame:
            self._update_special_endgame_state(b_)
        return b_

    def check_game_over(self, board, player):
        if self._special_endgame_draw:
            return 0
        if not self.get_valid_moves(board, player):
            return -1
        if not np.any(np.sign(board) == -player):
            return 1
        return None

    def reset_special_endgame_state(self):
        self._special_endgame_key = None
        self._special_endgame_halfmoves = 0
        self._special_endgame_draw = False

    def _is_on_big_diagonal(self, r, c):
        return r == c or (r + c) == (BOARD_SIZE - 1)

    def _collect_side_state(self, board, side):
        king_positions = np.argwhere(board == (2 * side))
        return {
            "kings": int(len(king_positions)),
            "men": int(np.sum(board == side)),
            "king_positions": [tuple(pos) for pos in king_positions]
        }

    def _resolve_special_endgame_key(self, board):
        white = self._collect_side_state(board, 1)
        black = self._collect_side_state(board, -1)
        sides = [("w", white), ("b", black)]
        for left_name, left in sides:
            right_name = "b" if left_name == "w" else "w"
            right = black if right_name == "b" else white
            if left["kings"] == 2 and left["men"] == 0 and right["kings"] == 2 and right["men"] == 0:
                return "2K_v_2K"
            if left["kings"] == 2 and left["men"] == 0 and right["kings"] == 1 and right["men"] == 0:
                return "2K_v_1K"
            if left["kings"] == 1 and left["men"] == 1 and right["kings"] == 1 and right["men"] == 0:
                return "1K1M_v_1K"
            if left["kings"] == 1 and left["men"] == 0 and right["kings"] == 1 and right["men"] == 1:
                return "1K_v_1K1M"
            if left["kings"] == 3 and left["men"] == 0 and right["kings"] == 1 and right["men"] == 0:
                lone_king_pos = right["king_positions"][0]
                if self._is_on_big_diagonal(lone_king_pos[0], lone_king_pos[1]):
                    return "3K_v_1K_lone_on_big_diagonal"
            if left["kings"] == 2 and left["men"] == 1 and right["kings"] == 1 and right["men"] == 0:
                return "2K1M_v_1K"
            if left["kings"] == 1 and left["men"] == 2 and right["kings"] == 1 and right["men"] == 0:
                return "1K2M_v_1K"
        if white["kings"] == 1 and white["men"] == 0 and black["kings"] == 1 and black["men"] == 0:
            return "1K_v_1K"
        return None

    def _update_special_endgame_state(self, board):
        key = self._resolve_special_endgame_key(board)
        if key is None:
            self.reset_special_endgame_state()
            return
        if key != self._special_endgame_key:
            self._special_endgame_key = key
            self._special_endgame_halfmoves = 1
        else:
            self._special_endgame_halfmoves += 1
        self._special_endgame_draw = self._special_endgame_halfmoves >= SPECIAL_ENDGAME_HALF_MOVES_LIMIT
