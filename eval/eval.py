--- START OF FILE eval.py ---

# Test script to challenge the trained Checkers Model.
# It loads the model and allows you to play against it locally.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from game import BOARD_SIZE, Checkers

# --- DEFINITIONS IDENTICAL TO TRAINING SCRIPT ---
# It is crucial that the network architecture and game logic are exactly the same.

DEVICE = torch.device("cpu") # Running on CPU is sufficient for inference.

# --- AI Components (Copied from training) ---
def state_to_tensor(board, player):
    tensor = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tensor[0, board == player] = 1; tensor[1, board == player*2] = 1
    tensor[2, board == -player] = 1; tensor[3, board == -player*2] = 1
    if player == 1: tensor[4,:,:] = 1.0
    return torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 64
        self.body = nn.Sequential(nn.Conv2d(5, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                                  nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                                  nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Conv2d(num_channels, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(),
                                         nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE))
        self.value_head = nn.Sequential(nn.Conv2d(num_channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                        nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(),
                                        nn.Linear(64, 1), nn.Tanh())
    def forward(self, x):
        x = self.body(x); return self.policy_head(x), self.value_head(x)

class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent; self.prior = prior; self.children = {}; self.visits = 0; self.value_sum = 0.0
    def get_value(self): return self.value_sum / self.visits if self.visits > 0 else 0.0

class MCTS:
    def __init__(self, game, model, sims=100, c_puct=1.5):
        self.game, self.model, self.sims, self.c_puct = game, model, sims, c_puct
    def run(self, board, player):
        root = MCTSNode()
        self._expand_and_evaluate(root, board, player)
        for _ in range(self.sims):
            node, search_board, search_player = root, np.copy(board), player
            search_path = [root]
            while node.children:
                move, node = self._select_child(node)
                search_board = self.game.apply_move(search_board, move, track_special_endgame=False); search_player *= -1; search_path.append(node)
            value = self.game.check_game_over(search_board, search_player)
            if value is None and node.visits == 0: value = self._expand_and_evaluate(node, search_board, search_player)
            elif value is None: value = node.get_value()
            for n in reversed(search_path): n.visits += 1; n.value_sum += value; value *= -1
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves])
        return moves, visits / (np.sum(visits) + 1e-9)
    def _select_child(self, node):
        sqrt_total_visits = np.sqrt(node.visits); best_move, max_score = None, -np.inf
        for move, child in node.children.items():
            score = -child.get_value() + self.c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
            if score > max_score: max_score, best_move = score, move
        return best_move, node.children[best_move]
    def _expand_and_evaluate(self, node, board, player):
        valid_moves = self.game.get_valid_moves(board, player)
        if not valid_moves: return -1.0
        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_to_tensor(board, player))
        value = value_tensor.item()
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        move_priors = {}; total_prior = 0
        for move in valid_moves:
            if isinstance(move, (list, tuple)) and len(move) > 0 and isinstance(move[0], tuple) and len(move[0]) == 2 and isinstance(move[0][0], tuple):
                start_pos_tuple = move[0][0]
            else:
                start_pos_tuple = move[0]
            start_pos_idx = start_pos_tuple[0] * BOARD_SIZE + start_pos_tuple[1]
            prior = policy_probs[start_pos_idx]
            key = tuple(move) if isinstance(move, list) else move
            move_priors[key] = prior; total_prior += prior
        if total_prior > 0:
            for move_key, prior in move_priors.items(): node.children[move_key] = MCTSNode(parent=node, prior=prior / total_prior)
        else:
            for move in valid_moves:
                key = tuple(move) if isinstance(move, list) else move
                node.children[key] = MCTSNode(parent=node, prior=1.0 / len(valid_moves))
        return value

# --- MAIN FUNCTION TO PLAY THE GAME ---
def play_game(model_path):
    print("--- Loading Checkers Master... ---")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found. Make sure it is in the same folder as this script.")
        return
        
    model = PolicyValueNetwork().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded. You (X) play first.")

    game = Checkers()
    mcts = MCTS(game, model)

    while True:
        board, player = game.get_initial_board(), 1 # Player 1 is Human (white pieces, 'x' and 'X')
        
        def print_board(b):
            chars = {1: 'x', 2: 'X', -1: 'o', -2: 'O', 0: '.'}
            print("\n  0 1 2 3 4 5 6 7 (cols)")
            for r_idx, row in enumerate(b):
                print(f"{r_idx} {' '.join(chars[val] for val in row)}")
            print()

        while game.check_game_over(board, player) is None:
            print_board(board)
            if player == 1:
                valid_moves = game.get_valid_moves(board, player)
                print("Your valid moves:")
                for i, move in enumerate(valid_moves): print(f"  {i}: {move}")
                
                while True:
                    try:
                        move_idx = int(input(f"Choose your move (0-{len(valid_moves)-1}): "))
                        if 0 <= move_idx < len(valid_moves):
                            move = valid_moves[move_idx]
                            break
                        else: print("Invalid index.")
                    except ValueError:
                        print("Please enter a number.")
            else:
                print("AI's turn... (thinking)")
                valid_moves, policy = mcts.run(np.copy(board), player)
                move = valid_moves[np.argmax(policy)]
                print(f"AI plays: {move}")

            board = game.apply_move(board, move)
            player *= -1

        print("\n--- GAME OVER ---")
        print_board(board)
        result = game.check_game_over(board, 1)
        if result == 1: print("YOU WON! Congratulations!")
        elif result == -1: print("The AI won.")
        else: print("Draw.")

        if input("\nPlay again? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    play_game("checkers_master_final.pth")