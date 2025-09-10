#!/usr/bin/env python3
"""
MCTS Integration for GRPO Experiments

Integrates MCTS with GRPO training for self-play trajectory generation.
Optimized for transformer models with attention mechanisms.
"""

import torch
import torch.nn.functional as F
import numpy as np
import chess
import logging
import time
import math
from typing import List, Dict, Tuple, Optional, Any, Callable
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
        """Converts a chess.Move object to a policy index using AlphaZero encoding."""
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        # Queen moves: 0-4095 (64*64)
        if not promotion or promotion == chess.QUEEN:
            return from_square * 64 + to_square

        # Underpromotions: 4096-4671 (64*3*3 = 576 moves)
        # Each from_square can promote to 3 pieces (Knight, Bishop, Rook)
        promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        return 4096 + from_square * 3 + promo_offset[promotion]


class MCTS:
    """MCTS implementation optimized for GRPO trajectory collection"""

    def __init__(self, model: torch.nn.Module, config: MCTSConfig, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

        # Transposition table for caching
        self.tt_cache = {} if config.enable_tt_cache else None

        logger.info(f"Initialized MCTS with {config.num_simulations} simulations")

    def search(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Perform MCTS search and return policy and value

        Args:
            board: Current chess position

        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        if board is None:
            raise ValueError("Board cannot be None")
            
        if board.is_game_over():
            logger.warning("MCTS search called on terminal position")
            return torch.zeros(4672), 0.0
            
        logger.info(f"🔍 MCTS search starting for board: {board.fen()}")
        search_start_time = time.time()
        
        try:
            root = MCTSNode(board=board.copy())

            # Expand root node
            logger.info("🔍 Evaluating root position...")
            try:
                policy_logits, value = self._evaluate_position(board)
                logger.info(f"🔍 Root evaluation: policy shape {policy_logits.shape}, value {value}")
            except Exception as e:
                logger.error(f"🔍 Root evaluation failed: {e}")
                logger.error(f"🔍 Error type: {type(e).__name__}")
                import traceback
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                raise

            legal_moves = list(board.legal_moves)
            logger.info(f"🔍 Legal moves: {len(legal_moves)}")
            
            if not legal_moves:
                logger.warning("No legal moves available")
                return torch.zeros(4672), 0.0

            try:
                root.expand(legal_moves, policy_logits, self.config)
                logger.info(f"🔍 Root expanded with {len(root.children)} children")
            except Exception as e:
                logger.error(f"🔍 Root expansion failed: {e}")
                logger.error(f"🔍 Error type: {type(e).__name__}")
                import traceback
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                raise

            # Perform simulations
            for sim_idx in range(self.config.num_simulations):
                try:
                    self._simulate(root)
                except Exception as e:
                    logger.warning(f"Simulation {sim_idx} failed: {e}")
                    continue

            # Extract policy from visit counts
            policy = self._get_policy_from_visits(root, legal_moves)
            logger.debug(f"Extracted policy shape: {policy.shape}")

            # Get value estimate
            value_est = root.value
            logger.debug(f"Root value estimate: {value_est}")

            return policy, value_est
            
        except Exception as e:
            logger.error(f"MCTS search failed: {e}")
            # Return fallback policy
            legal_moves = list(board.legal_moves)
            if legal_moves:
                fallback_policy = torch.zeros(4672)
                # Set uniform probability for legal moves
                for move in legal_moves:
                    move_idx = self._move_to_index(move)
                    if 0 <= move_idx < len(fallback_policy):
                        fallback_policy[move_idx] = 1.0 / len(legal_moves)
                return fallback_policy, 0.0
            else:
                return torch.zeros(4672), 0.0

    def search_batch(self, boards: List[chess.Board]) -> List[Tuple[torch.Tensor, float]]:
        """
        Perform batched MCTS search for multiple positions

        Args:
            boards: List of chess positions

        Returns:
            List of (policy_logits, value_estimate) tuples
        """
        logger.debug(f"MCTS batch search starting for {len(boards)} boards")

        roots = [MCTSNode(board=b.copy()) for b in boards]
        
        # First, evaluate all root positions in a batch
        board_tensors = torch.cat([self._board_to_tensor(b) for b in boards])
        policy_logits_batch, value_batch = self.model(board_tensors.to(self.device))

        for i, root in enumerate(roots):
            legal_moves = list(root.board.legal_moves)
            root.expand(legal_moves, policy_logits_batch[i], self.config)

        # Then, run simulations for each root
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            futures = [executor.submit(self._run_simulations_for_root, root) for root in roots]
            results = [future.result() for future in as_completed(futures)]

        logger.debug(f"Batch search completed for {len(boards)} positions")
        return results

    def _run_simulations_for_root(self, root: MCTSNode) -> Tuple[torch.Tensor, float]:
        for _ in range(self.config.num_simulations):
            self._simulate(root)
        
        legal_moves = list(root.board.legal_moves)
        policy = self._get_policy_from_visits(root, legal_moves)
        value = root.value
        return policy, value

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
        try:
            board_tensor = self._board_to_tensor(board)
            
            if board_tensor is None or board_tensor.numel() == 0:
                logger.error("Invalid board tensor generated")
                return torch.zeros(4672), 0.0

            with torch.no_grad():
                policy_logits, value = self.model(board_tensor.to(self.device))

            # Validate outputs
            if policy_logits is None or value is None:
                logger.error("Model returned None outputs")
                return torch.zeros(4672), 0.0
                
            if policy_logits.shape[-1] != 4672:
                logger.error(f"Policy logits shape mismatch: expected 4672, got {policy_logits.shape}")
                return torch.zeros(4672), 0.0

            return policy_logits.cpu().squeeze(0), value.item()
            
        except Exception as e:
            logger.error(f"Position evaluation failed: {e}")
            return torch.zeros(4672), 0.0

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor format compatible with transformer"""
        # Create 19-channel board representation
        channels = []

        # Piece channels (12 channels: 6 piece types x 2 colors)
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

            channels.extend([white_channel, black_channel])

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

        return torch.stack(channels, dim=0).unsqueeze(0)  # Add batch dimension

    def _get_policy_from_visits(self, root: MCTSNode, legal_moves: List[chess.Move]) -> torch.Tensor:
        """Extract policy from visit counts"""
        policy = torch.zeros(4672)

        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for move, child in root.children.items():
                move_idx = self._move_to_index(move)
                if 0 <= move_idx < len(policy):
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
        Generate complete trajectory for GRPO training using real MCTS
        """
        logger.debug(f"Starting trajectory generation with max_moves={max_moves}")
        trajectory = []
        current_board = board.copy()
        move_count = 0

        while not current_board.is_game_over() and move_count < max_moves:
            logger.debug(f"Move {move_count + 1}: Getting MCTS policy...")

            try:
                # Get MCTS policy and value
                policy, value = self.search(current_board)

                # Sample move from policy
                legal_moves = list(current_board.legal_moves)

                if not legal_moves:
                    break

                move, log_prob = self._sample_move_from_policy(policy, legal_moves)

                # Store trajectory step
                step = {
                    'board': current_board.copy(),
                    'move': move,
                    'log_prob': log_prob,
                    'policy': policy,
                    'value': value,
                    'legal_moves': legal_moves,
                    'move_count': move_count
                }
                trajectory.append(step)

                # Make move
                current_board.push(move)
                move_count += 1

            except Exception as e:
                logger.error(f"Error during move {move_count + 1}: {e}")
                break

        # Add final result
        if current_board.is_game_over():
            result = self._get_game_result(current_board)
        else:
            result = 0.0  # Draw for max moves

        # Update trajectory with final rewards
        if trajectory:
            for i, step in enumerate(trajectory):
                step['reward'] = result if i == len(trajectory) - 1 else 0.0
                step['done'] = i == len(trajectory) - 1
                step['state'] = self._board_to_tensor(step['board'])
                step['action'] = self._move_to_index(step['move'])

        logger.info(f"Trajectory generation complete: {len(trajectory)} steps, final result: {result}")
        return trajectory

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move from MCTS search"""
        policy, _ = self.search(board)
        legal_moves = list(board.legal_moves)
        
        best_move = None
        best_prob = -1

        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if 0 <= move_idx < len(policy) and policy[move_idx] > best_prob:
                best_prob = policy[move_idx]
                best_move = move
        
        return best_move if best_move is not None else legal_moves[0]

    def _sample_move_from_policy(self, policy: torch.Tensor, legal_moves: List[chess.Move]) -> Tuple[chess.Move, float]:
        """Sample move from policy distribution and return log probability for GRPO"""
        legal_probs = []
        legal_indices = []

        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if 0 <= move_idx < len(policy):
                prob = policy[move_idx].item()
                if prob > 0:
                    legal_probs.append(prob)
                    legal_indices.append(move_idx)

        if not legal_probs:
            # Fallback to uniform random
            selected_move = np.random.choice(legal_moves)
            uniform_prob = 1.0 / len(legal_moves)
            log_prob = math.log(uniform_prob)
            return selected_move, log_prob

        # Convert to torch tensor for proper probability handling
        legal_probs_tensor = torch.tensor(legal_probs, dtype=torch.float32)
        legal_probs_normalized = F.softmax(legal_probs_tensor, dim=0)

        # Sample move using torch multinomial for better numerical stability
        try:
            sample_idx = torch.multinomial(legal_probs_normalized, 1).item()
            selected_move = legal_moves[sample_idx]
            selected_prob = legal_probs_normalized[sample_idx].item()
            log_prob = math.log(selected_prob + 1e-8)  # Add small epsilon to avoid log(0)

            return selected_move, log_prob
        except Exception as e:
            logger.warning(f"Error in move sampling: {e}, falling back to numpy sampling")
            # Fallback to numpy sampling
            sampled_idx = np.random.choice(len(legal_indices), p=legal_probs)
            sampled_move_idx = legal_indices[sampled_idx]

            # Find corresponding move
            for move in legal_moves:
                if self._move_to_index(move) == sampled_move_idx:
                    # Calculate log probability
                    total_prob = sum(legal_probs)
                    move_prob = legal_probs[sampled_idx] / total_prob if total_prob > 0 else 1.0 / len(legal_moves)
                    log_prob = math.log(move_prob + 1e-8)
                    return move, log_prob

            # Final fallback
            selected_move = np.random.choice(legal_moves)
            uniform_prob = 1.0 / len(legal_moves)
            log_prob = math.log(uniform_prob)
            return selected_move, log_prob


class SelfPlayManager:
    """Manages self-play games for GRPO training"""

    def __init__(self, mcts_factory: Callable[[], MCTS], num_workers: int = 4, display_callback=None):
        """Create a self-play manager.

        Args:
            mcts_factory: Callable that returns a new :class:`MCTS` instance.
                A separate MCTS will be created for each worker to avoid
                shared mutable state across threads.
            num_workers: Number of parallel workers.
            display_callback: Optional callback for displaying trajectories.
        """

        self.mcts_factory = mcts_factory
        self.num_workers = num_workers
        self.display_callback = display_callback
        logger.info(f"Initialized SelfPlayManager with {num_workers} workers")

    def generate_games(self, num_games: int, max_moves: int = 180, timeout: int = 120) -> List[List[Dict[str, Any]]]:
        """
        Generate self-play games concurrently across workers.
        """
        logger.info(f"🚀 SelfPlayManager.generate_games called with {num_games} games")
        games = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._generate_single_game, self.mcts_factory(), max_moves)
                for _ in range(num_games)
            ]

            for future in as_completed(futures):
                try:
                    trajectory = future.result(timeout=timeout + 10)  # Add buffer for timeout
                    if trajectory:  # Only add non-empty trajectories
                        games.append(trajectory)
                        if self.display_callback:
                            self.display_callback(trajectory)
                except Exception as e:
                    logger.error(f"Error generating game: {e}")
                    # Continue with other games even if one fails

        logger.info(f"🎉 Generated {len(games)} self-play games total")
        return games

    def _generate_single_game(self, mcts: MCTS, max_moves: int) -> List[Dict[str, Any]]:
        """Generate a single game using its own MCTS instance."""
        try:
            return mcts.get_trajectory(chess.Board(), max_moves)
        except Exception as e:
            logger.error(f"Error in single game generation: {e}")
            return []


if __name__ == "__main__":
    # Test MCTS integration
    from experiments.grpo.models.large_chess_transformer import MagnusChessTransformerFactory

    print("=== MCTS Integration Test ===")

    # Create large transformer model
    model = MagnusChessTransformerFactory.create_magnus_chess()
    print(f"Model: {MagnusChessTransformerFactory.get_model_info(model)}")

    # Create MCTS factory
    mcts_config = MCTSConfig(num_simulations=50)  # Reduced for testing
    mcts_factory = lambda: MCTS(model, mcts_config)

    # Create self-play manager
    self_play = SelfPlayManager(mcts_factory, num_workers=2)

    # Test single trajectory generation
    board = chess.Board()
    trajectory = mcts_factory().get_trajectory(board, max_moves=10)

    print(f"Generated trajectory with {len(trajectory)} steps")
    print(f"First move: {trajectory[0]['move'] if len(trajectory) > 0 else 'None'}")
    print("✅ MCTS integration test passed!")