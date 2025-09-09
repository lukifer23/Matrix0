#!/usr/bin/env python3
"""
MCTS Integration for GRPO Experiments

Integrates MCTS with GRPO training for self-play trajectory generation.
Optimized for transformer models with attention mechanisms.
"""

import torch
import numpy as np
import chess
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS in GRPO experiments"""
    num_simulations: int = 200
    cpuct: float = 2.2
    cpuct_start: float = 2.8
    cpuct_end: float = 1.8
    cpuct_plies: int = 40
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    virtual_loss: float = 2.0
    batch_size: int = 24
    max_children: int = 32
    enable_tt_cache: bool = True
    tt_capacity: int = 100000


@dataclass
class MCTSNode:
    """MCTS tree node optimized for GRPO trajectory collection"""
    board: chess.Board
    parent: Optional['MCTSNode'] = None
    children: Dict[chess.Move, 'MCTSNode'] = None
    move_from_parent: Optional[chess.Move] = None

    # MCTS values
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    # Virtual loss for parallelization
    virtual_loss_count: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    @property
    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0

    def select_child(self, config: MCTSConfig) -> Tuple[chess.Move, 'MCTSNode']:
        """Select best child using PUCT formula"""
        best_score = float('-inf')
        best_move = None
        best_child = None

        sqrt_parent_visits = np.sqrt(self.visit_count)

        for move, child in self.children.items():
            # PUCT score
            exploitation = child.value
            exploration = config.cpuct * child.prior * sqrt_parent_visits / (1 + child.visit_count + child.virtual_loss_count)

            # Virtual loss penalty
            virtual_penalty = config.virtual_loss * child.virtual_loss_count

            score = exploitation + exploration - virtual_penalty

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, legal_moves: List[chess.Move], policy_logits: torch.Tensor,
              config: MCTSConfig) -> None:
        """Expand node with legal moves and policy priors"""
        # Apply legal move masking
        legal_mask = torch.zeros(4672)  # Standard chess move space
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if move_idx < len(legal_mask):
                legal_mask[move_idx] = 1.0

        # Mask and normalize policy
        masked_policy = policy_logits * legal_mask
        policy_probs = torch.softmax(masked_policy, dim=-1)

        # Add Dirichlet noise for exploration
        if self.parent is None:  # Root node
            noise = torch.distributions.Dirichlet(
                config.dirichlet_alpha * torch.ones_like(policy_probs)
            ).sample()
            policy_probs = (1 - config.dirichlet_frac) * policy_probs + config.dirichlet_frac * noise

        # Create child nodes
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if move_idx < len(policy_probs):
                prior = policy_probs[move_idx].item()

                # Create child board
                child_board = self.board.copy()
                child_board.push(move)

                # Create child node
                child = MCTSNode(
                    board=child_board,
                    parent=self,
                    move_from_parent=move,
                    prior=prior
                )

                self.children[move] = child

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to policy index (simplified)"""
        # This is a simplified move encoding - in practice you'd want
        # a proper move encoder that handles all 4672 possible moves
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion or 0

        # Simple encoding: from_square * 64 + to_square + promotion_offset
        base_idx = from_sq * 64 + to_sq
        if promotion:
            # Add promotion piece offsets
            if promotion == chess.QUEEN:
                base_idx += 64 * 64 * 1
            elif promotion == chess.ROOK:
                base_idx += 64 * 64 * 2
            elif promotion == chess.BISHOP:
                base_idx += 64 * 64 * 3
            elif promotion == chess.KNIGHT:
                base_idx += 64 * 64 * 4

        return min(base_idx, 4671)  # Cap at max index


class MCTS:
    """MCTS implementation optimized for GRPO trajectory collection"""

    def __init__(self, model: torch.nn.Module, config: MCTSConfig, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

        # Transposition table for caching
        self.tt_cache = {} if config.enable_tt_cache else None

        logger.info(f"Initialized MCTS with {config.num_simulations} simulations")

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to policy index (simplified)"""
        # This is a simplified move encoding - in practice you'd want
        # a proper move encoder that handles all 4672 possible moves
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion or 0

        # Simple encoding: from_square * 64 + to_square + promotion_offset
        base_idx = from_sq * 64 + to_sq
        if promotion:
            # Add promotion piece offsets
            if promotion == chess.QUEEN:
                base_idx += 64 * 64 * 1
            elif promotion == chess.ROOK:
                base_idx += 64 * 64 * 2
            elif promotion == chess.BISHOP:
                base_idx += 64 * 64 * 3
            elif promotion == chess.KNIGHT:
                base_idx += 64 * 64 * 4

        return min(base_idx, 4671)  # Cap at max index

    def search(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Perform MCTS search and return policy and value

        Args:
            board: Current chess position

        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        logger.debug(f"MCTS search starting for board: {board.fen()}")
        search_start_time = time.time()
        root = MCTSNode(board=board.copy())

        # Expand root node
        logger.debug("Evaluating root position...")
        policy_logits, value = self._evaluate_position(board)
        logger.debug(f"Root evaluation: policy shape {policy_logits.shape}, value {value}")

        legal_moves = list(board.legal_moves)
        logger.debug(f"Legal moves: {len(legal_moves)}")

        root.expand(legal_moves, policy_logits, self.config)
        logger.debug(f"Root expanded with {len(root.children)} children")

        # Perform simulations with reduced count for faster iteration
        sims_to_run = min(self.config.num_simulations, 25)  # Reduced further for speed
        logger.debug(f"Starting {sims_to_run} simulations...")
        for sim_idx in range(sims_to_run):
            self._simulate(root)

        # Extract policy from visit counts
        policy = self._get_policy_from_visits(root, legal_moves)
        logger.debug(f"Extracted policy shape: {policy.shape}")

        # Get value estimate
        value_est = root.value
        logger.debug(f"Root value estimate: {value_est}")

        return policy, value_est

    def search_batch(self, boards: List[chess.Board]) -> List[Tuple[torch.Tensor, float]]:
        """
        Perform batched MCTS search for multiple positions

        Args:
            boards: List of chess positions

        Returns:
            List of (policy_logits, value_estimate) tuples
        """
        logger.debug(f"MCTS batch search starting for {len(boards)} boards")

        results = []
        for board in boards:
            result = self.search(board)
            results.append(result)

        logger.debug(f"Batch search completed for {len(boards)} positions")
        return results

    def _simulate(self, root: MCTSNode) -> None:
        """Single MCTS simulation"""
        path = []
        current = root

        # Selection phase
        while current.is_expanded and not current.board.is_game_over():
            move, current = current.select_child(self.config)
            if current is None:
                break
            path.append(current)

        # Expansion phase
        if not current.board.is_game_over() and not current.is_expanded:
            policy_logits, _ = self._evaluate_position(current.board)
            legal_moves = list(current.board.legal_moves)
            current.expand(legal_moves, policy_logits, self.config)

            # Select first child for evaluation
            if current.children:
                first_child = next(iter(current.children.values()))
                path.append(first_child)
                current = first_child

        # Evaluation phase
        if current.board.is_game_over():
            result = self._get_game_result(current.board)
        else:
            _, value = self._evaluate_position(current.board)
            result = value

        # Backpropagation phase
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += result
            result = -result  # Flip for opponent

    def _evaluate_position(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """Evaluate position using neural network"""
        # Convert board to tensor format
        board_tensor = self._board_to_tensor(board)

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor.to(self.device))

        return policy_logits.cpu(), value.item()

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor format compatible with transformer"""
        # Create 19-channel board representation
        channels = []

        # Piece channels (12 channels: 6 piece types x 2 colors)
        piece_channels = []
        for piece_type in range(1, 7):  # 1-6: pawn, knight, bishop, rook, queen, king
            white_channel = torch.zeros(8, 8)
            black_channel = torch.zeros(8, 8)

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type:
                    row, col = divmod(square, 8)
                    if piece.color == chess.WHITE:
                        white_channel[row, col] = 1.0
                    else:
                        black_channel[row, col] = 1.0

            piece_channels.extend([white_channel, black_channel])

        channels.extend(piece_channels)

        # Additional channels for game state (7 more to make 19 total)
        # Side to move
        side_to_move = torch.ones(8, 8) if board.turn == chess.WHITE else torch.zeros(8, 8)
        channels.append(side_to_move)

        # Castling rights (4 channels)
        castling_channels = []
        for i in range(4):
            castling_channels.append(torch.zeros(8, 8))
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_channels[0].fill_(1.0)
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_channels[1].fill_(1.0)
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_channels[2].fill_(1.0)
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_channels[3].fill_(1.0)
        channels.extend(castling_channels)

        # En passant (1 channel)
        en_passant = torch.zeros(8, 8)
        if board.ep_square:
            row, col = divmod(board.ep_square, 8)
            en_passant[row, col] = 1.0
        channels.append(en_passant)

        # Halfmove clock (1 channel)
        halfmove = torch.full((8, 8), min(board.halfmove_clock / 100.0, 1.0))
        channels.append(halfmove)

        # Ensure all channels have the same shape
        processed_channels = []
        for channel in channels:
            if channel.dim() == 1:
                # If it's 1D, broadcast to 8x8
                channel = channel.unsqueeze(0).unsqueeze(0).expand(1, 8, 8).squeeze(0)
            processed_channels.append(channel)

        return torch.stack(processed_channels, dim=0).unsqueeze(0)  # Add batch dimension

    def _get_policy_from_visits(self, root: MCTSNode, legal_moves: List[chess.Move]) -> torch.Tensor:
        """Extract policy from visit counts"""
        policy = torch.zeros(4672)

        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for move, child in root.children.items():
                move_idx = root._move_to_index(move)
                if move_idx < len(policy):
                    policy[move_idx] = child.visit_count / total_visits

        return policy

    def _get_game_result(self, board: chess.Board) -> float:
        """Get game result from terminal position"""
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0.0
        else:
            # Should not happen in a properly terminated game
            return 0.0

    def get_trajectory(self, board: chess.Board, max_moves: int = 180) -> List[Dict[str, Any]]:
        """
        Generate trajectory for GRPO training

        Args:
            board: Starting position
            max_moves: Maximum moves to generate

        Returns:
            List of trajectory steps
        """
        logger.debug(f"Starting trajectory generation with max_moves={max_moves}")
        trajectory = []
        current_board = board.copy()
        move_count = 0

        logger.debug(f"Initial board FEN: {current_board.fen()}")
        logger.debug(f"Initial legal moves: {len(list(current_board.legal_moves))}")

        while not current_board.is_game_over() and move_count < max_moves:
            logger.debug(f"Move {move_count + 1}: Getting MCTS policy...")

            try:
                # Get MCTS policy and value
                policy, value = self.search(current_board)
                logger.debug(f"MCTS returned policy shape: {policy.shape}, value: {value}")

                # Sample move from policy
                legal_moves = list(current_board.legal_moves)
                logger.debug(f"Legal moves available: {len(legal_moves)}")

                if not legal_moves:
                    logger.debug("No legal moves available, ending game")
                    break

                move = self._sample_move_from_policy(policy, legal_moves)
                logger.debug(f"Selected move: {move}")

                # Store trajectory step
                step = {
                    'board': current_board.copy(),
                    'move': move,
                    'policy': policy,
                    'value': value,
                    'legal_moves': legal_moves,
                    'move_count': move_count
                }
                trajectory.append(step)

                # Make move
                current_board.push(move)
                move_count += 1
                logger.debug(f"Move made. New FEN: {current_board.fen()}")

            except Exception as e:
                logger.error(f"Error during move {move_count + 1}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                break

        # Add final result
        if current_board.is_game_over():
            result = self._get_game_result(current_board)
            logger.info(f"Game ended with result: {result}")
        else:
            result = 0.0  # Draw for max moves
            logger.info(f"Game reached max moves ({max_moves}), ending with draw")

        trajectory.append({
            'board': current_board.copy(),
            'result': result,
            'final': True
        })

        logger.info(f"Trajectory generation complete: {len(trajectory)} steps")
        return trajectory

    def _sample_move_from_policy(self, policy: torch.Tensor, legal_moves: List[chess.Move]) -> chess.Move:
        """Sample move from policy distribution"""
        # Get probabilities for legal moves
        legal_probs = []
        legal_indices = []

        for move in legal_moves:
            move_idx = self._move_to_index(move)  # Use the proper move-to-index method
            if move_idx < len(policy):
                prob = policy[move_idx].item()
                if prob > 0:
                    legal_probs.append(prob)
                    legal_indices.append(move_idx)

        if not legal_probs:
            # Fallback to uniform random
            return np.random.choice(legal_moves)

        # Normalize probabilities
        total_prob = sum(legal_probs)
        if total_prob > 0:
            legal_probs = [p / total_prob for p in legal_probs]

        # Sample move
        sampled_idx = np.random.choice(len(legal_indices), p=legal_probs)
        sampled_move_idx = legal_indices[sampled_idx]

        # Find corresponding move
        for move in legal_moves:
            if self._move_to_index(move) == sampled_move_idx:
                return move

        # Fallback
        return np.random.choice(legal_moves)


class SelfPlayManager:
    """Manages self-play games for GRPO training"""

    def __init__(self, mcts: MCTS, num_workers: int = 4, display_callback=None):
        self.mcts = mcts
        self.num_workers = num_workers
        self.display_callback = display_callback
        logger.info(f"Initialized SelfPlayManager with {num_workers} workers")

    def generate_games(self, num_games: int, max_moves: int = 180) -> List[List[Dict[str, Any]]]:
        """
        Generate self-play games using sequential processing for debugging

        Args:
            num_games: Number of games to generate
            max_moves: Maximum moves per game

        Returns:
            List of game trajectories
        """
        logger.info(f"🚀 SelfPlayManager.generate_games called with {num_games} games")
        logger.info(f"🎯 MCTS config: sims={self.mcts.config.num_simulations}, device={self.mcts.device}")
        games = []

        # For debugging, let's try sequential generation first
        logger.info("🔄 Starting sequential game generation...")

        for game_idx in range(num_games):
            logger.info(f"🎮 Starting game {game_idx + 1}/{num_games}")
            try:
                board = chess.Board()  # Start from standard position
                logger.info(f"📋 Initial board: {board.fen()}")
                logger.info(f"♟️  Legal moves: {len(list(board.legal_moves))}")

                trajectory = self.mcts.get_trajectory(board, max_moves)
                games.append(trajectory)
                logger.info(f"✅ Generated game {game_idx + 1} with {len(trajectory)} steps")

                # Update display in real-time
                if self.display_callback:
                    logger.debug(f"Calling display callback with trajectory of {len(trajectory)} steps")
                    self.display_callback(trajectory)
                else:
                    logger.warning("No display callback set!")
            except Exception as e:
                logger.error(f"❌ Error generating game {game_idx + 1}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info(f"🎉 Generated {len(games)} self-play games total")
        return games


if __name__ == "__main__":
    # Test MCTS integration
    from experiments.grpo.models.large_chess_transformer import LargeChessTransformerFactory

    print("=== MCTS Integration Test ===")

    # Create large transformer model
    model = LargeChessTransformerFactory.create_large()
    print(f"Model: {LargeChessTransformerFactory.get_model_info(model)}")

    # Create MCTS
    mcts_config = MCTSConfig(num_simulations=50)  # Reduced for testing
    mcts = MCTS(model, mcts_config)

    # Create self-play manager
    self_play = SelfPlayManager(mcts, num_workers=2)

    # Test single trajectory generation
    board = chess.Board()
    trajectory = mcts.get_trajectory(board, max_moves=10)

    print(f"Generated trajectory with {len(trajectory)} steps")
    print(f"First move: {trajectory[0]['move'] if len(trajectory) > 0 else 'None'}")
    print("✅ MCTS integration test passed!")
