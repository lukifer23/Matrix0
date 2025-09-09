#!/usr/bin/env python3
"""
GRPO Chess Orchestrator

Matrix0-style orchestrator for GRPO experiments with:
- TUI tables showing W/L/D, avg ms/move, sims, etc.
- Heartbeat monitoring
- Self-play → Training → Evaluation cycle
- Professional logging and metrics
"""

import argparse
import copy
import logging
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import yaml

import torch
import torch.nn as nn
import chess
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns

# Add the experiments directory to Python path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

from models.large_chess_transformer import MagnusChessTransformerFactory
from training.grpo_trainer import GRPOTrainer, GRPOConfig, Trajectory, TrajectoryStep
from mcts.mcts_integration import MCTS, MCTSConfig, SelfPlayManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-7s %(message)s',
    handlers=[
        logging.FileHandler('logs/grpo_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GRPOOrchestrator:
    """Matrix0-style orchestrator for GRPO experiments"""

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        self.config = config
        self.device = device
        self.console = Console()

        # Initialize components
        self.model = None
        self.mcts = None
        self.grpo_trainer = None
        self.self_play_manager = None

        # Experiment state
        self.epoch = 0
        self.total_games = 0
        self.start_time = datetime.now()

        # Metrics tracking
        self.metrics = {
            'self_play': [],
            'training': [],
            'evaluation': [],
            'performance': {
                'total_games': 0,
                'win_rate': 0.0,
                'draw_rate': 0.0,
                'avg_game_length': 0.0,
                'avg_ms_per_move': 0.0,
                'games_per_second': 0.0
            }
        }

        # Live display
        self.live = None
        self.display_thread = None
        self.current_phase = "Initializing"

        # Store trajectories for display
        self.current_trajectories = []

        # Heartbeat
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 5  # seconds

        self._initialize_components()
        logger.info("GRPO Orchestrator initialized")

    def _convert_mcts_to_grpo_trajectories(self, mcts_trajectories):
        """
        Convert MCTS trajectory format (list of dicts) to GRPO Trajectory objects
        """
        grpo_trajectories = []

        for mcts_traj in mcts_trajectories:
            if not mcts_traj:
                continue

            # Convert each step to TrajectoryStep
            trajectory_steps = []
            total_reward = 0.0

            for step_dict in mcts_traj:
                state = step_dict.get('state')
                if state is None and 'board' in step_dict:
                    state = self._board_to_tensor(step_dict['board'])

                # Ensure state is properly formatted
                if state is not None:
                    if state.dim() == 3:  # (C, H, W)
                        state = state.unsqueeze(0)  # Add batch dimension
                    elif state.dim() == 4 and state.size(0) == 1:  # Already has batch dimension
                        pass
                    else:
                        logger.warning(f"Unexpected state shape: {state.shape}, using default")
                        state = torch.zeros(1, 19, 8, 8)
                else:
                    state = torch.zeros(1, 19, 8, 8)

                step = TrajectoryStep(
                    state=state,
                    action=step_dict.get('action', 0),
                    log_prob=step_dict.get('log_prob', 0.0),
                    value=step_dict.get('value', 0.0),
                    reward=step_dict.get('reward', 0.0),
                    done=step_dict.get('done', False),
                    legal_mask=step_dict.get('legal_mask')
                )
                trajectory_steps.append(step)
                total_reward += step.reward

            if trajectory_steps:
                game_result = 0.0
                if trajectory_steps:
                    final_reward = trajectory_steps[-1].reward
                    if final_reward > 0:
                        game_result = 1.0
                    elif final_reward < 0:
                        game_result = -1.0

                trajectory = Trajectory(
                    steps=trajectory_steps,
                    total_reward=total_reward,
                    length=len(trajectory_steps),
                    game_result=game_result
                )
                grpo_trajectories.append(trajectory)

        logger.info(f"Converted {len(mcts_trajectories)} MCTS trajectories to {len(grpo_trajectories)} GRPO trajectories")
        return grpo_trajectories

    def _board_to_tensor(self, board):
        """Convert chess board to tensor format (copied from MCTS)"""
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

        return torch.stack(channels, dim=0).unsqueeze(0)  # Add batch dimension

    def _initialize_components(self):
        """Initialize all experiment components"""
        # Load model
        model_config = self.config['model']
        model_type = model_config['type']

        if model_type in ['magnus_transformer', 'ultra_large_transformer']:
            logger.info("Creating MAGNUS transformer (~70M parameters)")
            self.model = MagnusChessTransformerFactory.create_magnus_chess()
        elif model_type == 'medium_transformer':
            logger.info("Creating MEDIUM transformer (~25M parameters)")
            self.model = MagnusChessTransformerFactory.create_medium_transformer()
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: magnus_transformer, medium_transformer")

        # Load checkpoint if exists
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.epoch = checkpoint.get('epoch', 0)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        self.model.to(self.device)

        # Initialize MCTS
        grpo_config = self.config['grpo']
        mcts_config = MCTSConfig(
            num_simulations=grpo_config.get('mcts_simulations', 150),
            cpuct=grpo_config.get('cpuct', 2.2),
            virtual_loss=grpo_config.get('virtual_loss', 2.0),
            batch_size=grpo_config.get('mcts_batch_size', 24)
        )
        self.mcts = MCTS(self.model, mcts_config, self.device)

        # Initialize GRPO trainer
        grpo_cfg = GRPOConfig(
            group_size=grpo_config.get('group_size', 4),
            clip_epsilon=grpo_config.get('clip_epsilon', 0.2),
            learning_rate=float(grpo_config.get('learning_rate', 8e-5)),
            batch_size=grpo_config.get('batch_size', 32)
        )
        self.grpo_trainer = GRPOTrainer(self.model, grpo_cfg, self.device)

        def display_callback(new_trajectory):
            self.current_trajectories.append(new_trajectory)
            self.total_games = len(self.current_trajectories)

        self.self_play_manager = SelfPlayManager(
            self.mcts,
            num_workers=grpo_config.get('num_workers', 3),
            display_callback=display_callback
        )

    def run_experiment(self, num_games: int = 30, max_epochs: int = 1):
        logger.info(f"Starting GRPO experiment: {num_games} games, {max_epochs} epochs")

        with Live(self._create_display_content(), refresh_per_second=0.5, console=self.console, transient=False) as live:
            self.live = live
            self._start_heartbeat()

            for epoch in range(max_epochs):
                self.epoch = epoch + 1
                self.current_trajectories = []
                self.total_games = 0

                # Phase 1: Self-play
                self.current_phase = "Self-Play"
                self._run_self_play_phase(num_games)

                # Phase 2: Training
                self.current_phase = "Training"
                self._run_training_phase()

                # Phase 3: Evaluation
                self.current_phase = "Evaluation"
                self._run_evaluation_phase()

                # Save checkpoint
                self._save_checkpoint()

            self._stop_heartbeat()

    def _run_self_play_phase(self, num_games: int):
        logger.info(f"🚀 Starting self-play: {num_games} games")
        start_time = time.time()
        
        try:
            trajectories = self.self_play_manager.generate_games(
                num_games, 
                max_moves=180, 
                timeout=self.config.get('mcts_timeout', 120)
            )
            self.current_trajectories = self._convert_mcts_to_grpo_trajectories(trajectories)
            
            if not self.current_trajectories:
                logger.error("No valid trajectories generated in self-play phase")
                return
                
        except Exception as e:
            logger.error(f"Error in self-play phase: {e}")
            self.current_trajectories = []
            return
            
        duration = time.time() - start_time

        self.metrics['self_play'].append({
            'epoch': self.epoch,
            'games_generated': len(self.current_trajectories),
            'duration_seconds': duration
        })
        
        logger.info(f"Self-play completed: {len(self.current_trajectories)} games in {duration:.2f}s")

    def _run_training_phase(self):
        logger.info("Starting GRPO training phase")
        if not self.current_trajectories:
            logger.warning("No trajectories to train on.")
            return

        try:
            training_metrics = self.grpo_trainer.train_on_trajectories(self.current_trajectories)
            self.metrics['training'].append(training_metrics)

            # Learning rate scheduling
            if training_metrics and training_metrics.get('policy_loss', 0) > 1.0:
                for param_group in self.grpo_trainer.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.warning(f"High policy loss, reducing learning rate to {self.grpo_trainer.optimizer.param_groups[0]['lr']}")
                
        except Exception as e:
            logger.error(f"Error in training phase: {e}")
            self.metrics['training'].append({'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0})

    def _run_evaluation_phase(self):
        logger.info("Starting evaluation phase")
        eval_games = self.config.get('training', {}).get('eval_games', 10)
        
        try:
            eval_results = self._run_evaluation_games(eval_games)
            self.metrics['evaluation'].append(eval_results)
        except Exception as e:
            logger.error(f"Error in evaluation phase: {e}")
            self.metrics['evaluation'].append({
                'trained_wins': 0,
                'baseline_wins': 0,
                'draws': 0,
                'win_rate': 0.0
            })

    def _run_evaluation_games(self, num_games: int) -> Dict[str, Any]:
        logger.info(f"Running {num_games} evaluation games")
        trained_wins = 0
        baseline_wins = 0
        draws = 0

        try:
            baseline_model = self._load_baseline_model()
            baseline_mcts = MCTS(baseline_model, self.mcts.config, self.device)

            for i in range(num_games):
                try:
                    result, _ = self._play_evaluation_game(self.mcts, baseline_mcts)
                    if result == 1:
                        trained_wins += 1
                    elif result == -1:
                        baseline_wins += 1
                    else:
                        draws += 1
                except Exception as e:
                    logger.warning(f"Error in evaluation game {i+1}: {e}")
                    draws += 1  # Count as draw if game fails

        except Exception as e:
            logger.error(f"Error setting up evaluation: {e}")
            return {
                'trained_wins': 0,
                'baseline_wins': 0,
                'draws': num_games,
                'win_rate': 0.0
            }

        return {
            'trained_wins': trained_wins,
            'baseline_wins': baseline_wins,
            'draws': draws,
            'win_rate': trained_wins / num_games if num_games > 0 else 0
        }

    def _load_baseline_model(self):
        baseline_path = Path(self.config.get('model', {}).get('baseline_checkpoint', 'checkpoints/baseline_medium_transformer_fresh.pt'))
        if not baseline_path.exists():
            logger.warning(f"Baseline checkpoint not found at {baseline_path}, using a copy of the current model.")
            return copy.deepcopy(self.model)

        checkpoint = torch.load(baseline_path, map_location=self.device)
        model_type = checkpoint['config']['model_type']

        if model_type in ['magnus_transformer', 'ultra_large_transformer']:
            baseline_model = MagnusChessTransformerFactory.create_magnus_chess()
        elif model_type == 'medium_transformer':
            baseline_model = MagnusChessTransformerFactory.create_medium_transformer()
        else:
            raise ValueError(f"Unsupported model type in baseline checkpoint: {model_type}. Supported types: magnus_transformer, medium_transformer")

        baseline_model.load_state_dict(checkpoint['model_state_dict'])
        baseline_model.to(self.device)
        baseline_model.eval()
        return baseline_model

    def _play_evaluation_game(self, trained_mcts: MCTS, baseline_mcts: MCTS) -> Tuple[int, int]:
        board = chess.Board()
        moves_played = 0
        max_moves = 200

        while not board.is_game_over() and moves_played < max_moves:
            if board.turn == chess.WHITE:
                move = trained_mcts.get_move(board)
            else:
                move = baseline_mcts.get_move(board)

            if move is None:
                break

            board.push(move)
            moves_played += 1

        if board.is_checkmate():
            return (1, moves_played) if board.turn == chess.BLACK else (-1, moves_played)
        else:
            return (0, moves_played)

    def _save_checkpoint(self):
        checkpoint_dir = Path("results/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"grpo_checkpoint_epoch_{self.epoch}.pt"
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.grpo_trainer.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _create_display_content(self) -> str:
        uptime = datetime.now() - self.start_time
        last_heartbeat = datetime.now() - self.last_heartbeat
        perf = self.metrics['performance']
        total_moves = sum(len(traj.steps) for traj in self.current_trajectories)

        return f"""
        🎯 GRPO Chess Experiment - Epoch {self.epoch}
        Started: {self.start_time.strftime('%H:%M:%S')} | Games: {self.total_games} | Moves: {total_moves} | Uptime: {str(uptime).split('.')[0]}

        📊 Performance Metrics
        ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
        ┃ Metric               ┃ Value   ┃ Target  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
        │ Win Rate             │ {perf['win_rate']:.3f}   │ ≥ 0.30  │
        │ Draw Rate            │ {perf['draw_rate']:.3f}   │ ≤ 0.30  │
        │ Avg Game Length      │ {perf['avg_game_length']:.1f}    │ 60-90   │
        │ Avg ms/move          │ {perf['avg_ms_per_move']:.1f}    │ ≤ 200   │
        │ Games/sec            │ {perf['games_per_second']:.2f}    │ ≥ 1.0   │
        └───────────────────────┴─────────┴─────────┘

        🔄 Status: 💓 {last_heartbeat.seconds}s ago | 🎮 {self.device} | 🧠 {self.config['model']['type']} | ⚡ {self.config['grpo']['mcts_simulations']} sims
        📍 Phase: {self.current_phase}
        """

    def _start_heartbeat(self):
        def heartbeat():
            while not self.stop_heartbeat.is_set():
                self.last_heartbeat = datetime.now()
                if self.live:
                    self.live.update(self._create_display_content())
                time.sleep(self.heartbeat_interval)

        self.stop_heartbeat = threading.Event()
        self.display_thread = threading.Thread(target=heartbeat, daemon=True)
        self.display_thread.start()

    def _stop_heartbeat(self):
        if self.display_thread:
            self.stop_heartbeat.set()
            self.display_thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description='GRPO Chess Orchestrator')
    parser.add_argument('--config', type=str, default='medium_transformer_grpo',
                       help='Experiment configuration name')
    parser.add_argument('--games', type=int, default=30,
                       help='Number of self-play games per epoch')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (cpu/cuda/mps)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')

    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / 'configs' / 'experiment_configs.yaml'
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if args.config not in configs:
        print(f"❌ Configuration '{args.config}' not found")
        sys.exit(1)

    config = configs[args.config]
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"🚀 Starting GRPO Orchestrator")
    print(f"   Config: {args.config}")
    print(f"   Games: {args.games}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Device: {device}")
    if config.get('checkpoint_path'):
        print(f"   Checkpoint: {config['checkpoint_path']}")
    print()

    orchestrator = GRPOOrchestrator(config, device)
    orchestrator.run_experiment(args.games, args.epochs)


if __name__ == "__main__":
    main()
