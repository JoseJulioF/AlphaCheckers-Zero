import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
import random
import time
import os
import argparse
from game import BOARD_SIZE, Checkers

# --- Seeds for Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- OPTIMIZED Configurations for T4/P100 ---
NUM_EPISODES = 500
SELF_PLAY_GAMES = 25
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
REPLAY_BUFFER_SIZE = 75000
MCTS_SIMS = 80
C_PUCT = 2.0
CHECKPOINT_FILE = "checkers_checkpoint.pth"
SAVE_INTERVAL = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Optimized Device: {DEVICE}")

# --- Data Components and Neural Network ---
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

# --- MCTS ---
class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent; self.prior = prior; self.children = {}; self.visits = 0; self.value_sum = 0.0
    def get_value(self): return self.value_sum / self.visits if self.visits > 0 else 0.0
class MCTS:
    def __init__(self, game, model): self.game, self.model = game, model
    def run(self, board, player):
        root = MCTSNode()
        self._expand_and_evaluate(root, board, player)
        for _ in range(MCTS_SIMS):
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
            score = -child.get_value() + C_PUCT * child.prior * sqrt_total_visits / (1 + child.visits)
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

# --- Dataset and Training Function ---
class CheckersDataset(Dataset):
    def __init__(self, buffer): self.buffer = buffer
    def __len__(self): return len(self.buffer)
    def __getitem__(self, idx):
        board, player, policy_map, value = self.buffer[idx]
        policy_tensor = torch.zeros(BOARD_SIZE * BOARD_SIZE)
        total_prob = 0
        for move, prob in policy_map.items():
            if isinstance(move, (list, tuple)) and len(move) > 0 and isinstance(move[0], tuple) and len(move[0]) == 2 and isinstance(move[0][0], tuple):
                start_pos_tuple = move[0][0]
            else:
                start_pos_tuple = move[0]
            start_idx = start_pos_tuple[0] * BOARD_SIZE + start_pos_tuple[1]
            policy_tensor[start_idx] += prob; total_prob += prob
        if total_prob > 0: policy_tensor /= total_prob
        return state_to_tensor(board, player).squeeze(0), policy_tensor, torch.tensor([value], dtype=torch.float32)
def train_net(model, optimizer, dataloader):
    model.train()
    total_loss_p, total_loss_v = 0, 0
    for boards, policies, values in dataloader:
        boards, policies, values = boards.to(DEVICE), policies.to(DEVICE), values.to(DEVICE)
        p_logits, v_preds = model(boards)
        loss_p = F.cross_entropy(p_logits, policies); loss_v = F.mse_loss(v_preds.squeeze(), values.squeeze())
        loss = loss_p + loss_v
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss_p += loss_p.item(); total_loss_v += loss_v.item()
    return total_loss_p / len(dataloader), total_loss_v / len(dataloader)

# --- Checkpoint Functions ---
def save_checkpoint(state, filename):
    print("=> Saving checkpoint...")
    torch.save(state, filename)
    print("Checkpoint saved successfully.")
def load_checkpoint(filename, model, optimizer):
    if os.path.exists(filename):
        print("=> Loading checkpoint...")
        # ✅ FIX: Added weights_only=False to allow loading the replay_buffer (deque).
        checkpoint = torch.load(filename, weights_only=False)
        start_episode = checkpoint['episode']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        replay_buffer = checkpoint['replay_buffer']
        print(f"Checkpoint loaded. Resuming from episode {start_episode}.")
        return start_episode, model, optimizer, replay_buffer
    else:
        print("=> No checkpoint found. Starting from scratch.")
        return 0, model, optimizer, deque(maxlen=REPLAY_BUFFER_SIZE)

def run_training(num_episodes=NUM_EPISODES, self_play_games=SELF_PLAY_GAMES, max_moves=150, save_interval=SAVE_INTERVAL):
    model = PolicyValueNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    start_episode, model, optimizer, replay_buffer = load_checkpoint(CHECKPOINT_FILE, model, optimizer)
    if start_episode >= num_episodes:
        print(f"Nothing to run: checkpoint episode {start_episode} >= target episodes {num_episodes}.")
        return

    print("Starting training cycle... This may take many hours.")
    for episode in range(start_episode, num_episodes):
        start_time = time.time(); model.eval()
        for _ in range(self_play_games):
            game = Checkers()
            game_history = []; board, player = game.get_initial_board(), 1; move_count = 0
            while move_count < max_moves:
                result = game.check_game_over(board, player)
                if result is not None: final_value = -result; break
                mcts = MCTS(game, model); valid_moves, policy = mcts.run(board, player)
                if not valid_moves: final_value = -1.0; break
                policy_map = {m: p for m, p in zip(valid_moves, policy)}
                game_history.append((board, player, policy_map))
                move_idx = np.random.choice(len(valid_moves), p=policy)
                move = valid_moves[move_idx]; board = game.apply_move(board, move); player *= -1; move_count += 1
            else: final_value = 0.0
            for b, p, pol_map in game_history:
                value = final_value if p != player else -final_value
                replay_buffer.append((b, p, pol_map, value))
        if len(replay_buffer) > BATCH_SIZE * 10:
            dataset = CheckersDataset(list(random.sample(replay_buffer, BATCH_SIZE * 10)))
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            avg_loss_p, avg_loss_v = train_net(model, optimizer, dataloader)
            print(f"Ep. {episode+1}/{num_episodes} | Buffer: {len(replay_buffer)} | Time: {time.time()-start_time:.1f}s | Loss P: {avg_loss_p:.4f} | Loss V: {avg_loss_v:.4f}")
        else:
            print(f"Ep. {episode+1}/{num_episodes} | Buffer: {len(replay_buffer)} | Time: {time.time()-start_time:.1f}s | Collecting data...")
        if save_interval > 0 and (episode + 1) % save_interval == 0:
            state = {'episode': episode + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'replay_buffer': replay_buffer}
            save_checkpoint(state, CHECKPOINT_FILE)

    torch.save(model.state_dict(), "checkers_master_final.pth")
    print("\nTraining cycle completed.")


def main():
    parser = argparse.ArgumentParser(description="AlphaCheckers-Zero trainer")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Target total episodes")
    parser.add_argument("--self-play-games", type=int, default=SELF_PLAY_GAMES, help="Self-play games per episode")
    parser.add_argument("--max-moves", type=int, default=150, help="Max half-moves per self-play game")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL, help="Checkpoint save interval (0 disables)")
    args = parser.parse_args()
    run_training(
        num_episodes=args.episodes,
        self_play_games=args.self_play_games,
        max_moves=args.max_moves,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
