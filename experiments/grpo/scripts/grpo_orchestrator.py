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
import logging
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns

# Import our experiment components
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.large_chess_transformer import LargeChessTransformerFactory
    from training.grpo_trainer import GRPOTrainer, GRPOConfig
    from training.reward_shaping import ChessRewardShaper
    from mcts.mcts_integration import MCTS, MCTSConfig, SelfPlayManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the experiments/grpo directory")
    sys.exit(1)

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
        self.reward_shaper = None

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

    def _initialize_components(self):
        """Initialize all experiment components"""
        # Load model
        model_config = self.config['model']
        if model_config['type'] == 'medium_transformer':
            self.model = LargeChessTransformerFactory.create_medium_large()
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        # Load checkpoint if exists
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            # Load state dict with strict=False to handle parameter shape mismatches
            # (relative_pos_emb changed from [15,6,64] to [64,6,64])
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            # Log successful loading
            if not missing_keys and not unexpected_keys:
                logger.info("Model state_dict loaded successfully")
            else:
                logger.info("Model state_dict loaded with some mismatches (this is expected due to architecture updates)")

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

        # Initialize reward shaping if enabled
        if self.config.get('reward_shaping', {}).get('enabled', False):
            self.reward_shaper = ChessRewardShaper()

        # Create display update callback
        def display_callback(current_games):
            """Callback to update display with current game progress"""
            self.current_trajectories = current_games
            self.total_games = len(current_games)
            if hasattr(self, 'live'):
                self.live.update(self._create_display_content())

        # Initialize self-play manager with display callback
        self.self_play_manager = SelfPlayManager(
            self.mcts,
            num_workers=grpo_config.get('num_workers', 3),
            display_callback=display_callback
        )

    def run_experiment(self, num_games: int = 30, max_epochs: int = 1):
        """Run the complete experiment"""
        logger.info(f"Starting GRPO experiment: {num_games} games, {max_epochs} epochs")

        try:
            # Create initial display content
            initial_content = self._create_display_content()

            with Live(initial_content, refresh_per_second=0.5, console=self.console, transient=False) as live:
                self.live = live

                # Reduce logging level to avoid interfering with display
                old_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.INFO)  # Keep INFO level for debugging

                # Start heartbeat thread
                self._start_heartbeat()

                logger.info("Entering main experiment loop...")
                for epoch in range(max_epochs):
                    self.epoch = epoch + 1
                    logger.info(f"=== Epoch {self.epoch}/{max_epochs} ===")

                    # Update display for new epoch
                    live.update(self._create_display_content())

                    try:
                        # Phase 1: Self-play
                        self.current_phase = "Self-Play"
                        logger.info("About to start self-play phase...")
                        live.update(self._create_display_content())

                        logger.info("🔄 Calling _run_self_play_phase...")
                        self._run_self_play_phase(num_games)
                        logger.info("✅ Self-play phase completed")

                        # Update display after self-play
                        live.update(self._create_display_content())

                        # Phase 2: Training
                        self.current_phase = "Training"
                        logger.info("About to start training phase...")
                        live.update(self._create_display_content())

                        self._run_training_phase()
                        logger.info("Training phase completed")

                        # Update display after training
                        live.update(self._create_display_content())

                        # Phase 3: Evaluation
                        self.current_phase = "Evaluation"
                        logger.info("About to start evaluation phase...")
                        live.update(self._create_display_content())

                        self._run_evaluation_phase()
                        logger.info("Evaluation phase completed")

                        # Update display after evaluation
                        live.update(self._create_display_content())

                        # Save checkpoint after training
                        logger.info("About to save checkpoint...")
                        self._save_checkpoint()
                        logger.info("Checkpoint saved")

                        # Evaluate trained model vs baseline
                        logger.info("About to run evaluation...")
                        self._run_evaluation_phase()
                        logger.info("Evaluation completed")

                        # Final display update for epoch completion
                        live.update(self._create_display_content())

                        logger.info(f"✅ Epoch {self.epoch} completed successfully!")

                    except Exception as epoch_error:
                        logger.error(f"Error in epoch {self.epoch}: {epoch_error}")
                        logger.error(f"Error type: {type(epoch_error).__name__}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # Continue to next epoch rather than failing completely

                logger.info("🎉 Experiment completed successfully!")

        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user")
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(old_level)
            logger.info("Stopping heartbeat...")
            self._stop_heartbeat()
            logger.info("Experiment cleanup complete")

    def _run_self_play_phase(self, num_games: int):
        """Run self-play phase to generate training data"""
        logger.info(f"🚀 Starting self-play: {num_games} games")

        start_time = time.time()
        try:
            logger.info(f"📞 About to call self_play_manager.generate_games({num_games})")
            trajectories = self.self_play_manager.generate_games(num_games)
            logger.info(f"📦 Self-play manager returned {len(trajectories)} trajectories")

            # Store trajectories for display
            self.current_trajectories = trajectories

            # Process trajectories
            total_positions = sum(len(traj) for traj in trajectories)
            game_lengths = [len(traj) for traj in trajectories]

            # Calculate basic stats
            avg_game_length = np.mean(game_lengths) if game_lengths else 0
            duration = time.time() - start_time

            # Update metrics
            self.metrics['self_play'].append({
                'epoch': self.epoch,
                'games_generated': len(trajectories),
                'total_positions': total_positions,
                'avg_game_length': avg_game_length,
                'duration_seconds': duration,
                'games_per_second': len(trajectories) / duration if duration > 0 else 0
            })

            # Update performance metrics for display
            if trajectories:
                self.metrics['performance']['avg_game_length'] = avg_game_length
                self.metrics['performance']['games_per_second'] = len(trajectories) / duration if duration > 0 else 0

            self.total_games += len(trajectories)
            logger.info(f"Self-play complete: {len(trajectories)} games, {total_positions} positions")

            # Update display with new metrics
            if hasattr(self, 'live'):
                self.live.update(self._create_display_content())

        except Exception as e:
            logger.error(f"Error in self-play phase: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Create empty trajectories to continue
            trajectories = []
            self.metrics['self_play'].append({
                'epoch': self.epoch,
                'games_generated': 0,
                'total_positions': 0,
                'avg_game_length': 0,
                'duration_seconds': time.time() - start_time,
                'error': str(e)
            })

    def _run_training_phase(self):
        """Run GRPO training phase"""
        logger.info("Starting GRPO training phase")

        # Use actual trajectories from self-play
        if not self.current_trajectories:
            logger.warning("No trajectories available for training, using dummy data")
            trajectories = self._create_dummy_trajectories()
        else:
            trajectories = self.current_trajectories
            logger.info(f"Using {len(trajectories)} real trajectories for training")

        start_time = time.time()
        training_metrics = self.grpo_trainer.train_on_trajectories(trajectories)
        duration = time.time() - start_time

        # Update metrics
        training_metrics.update({
            'epoch': self.epoch,
            'duration_seconds': duration,
            'trajectories_used': len(trajectories)
        })

        self.metrics['training'].append(training_metrics)
        logger.info(f"Training complete: {training_metrics}")

    def _run_evaluation_phase(self):
        """Run evaluation phase"""
        logger.info("Starting evaluation phase")

        eval_games = self.config.get('training', {}).get('eval_games', 10)

        # Run evaluation games
        eval_results = self._run_evaluation_games(eval_games)

        # Update performance metrics
        self.metrics['performance'].update(eval_results)
        self.metrics['evaluation'].append(eval_results)

        logger.info(f"Evaluation: Trained {eval_results.get('trained_wins', 0)}W - {eval_results.get('baseline_wins', 0)}L - {eval_results.get('draws', 0)}D (Win rate: {eval_results['win_rate']:.3f})")

    def _run_evaluation_games(self, num_games: int) -> Dict[str, Any]:
        """Run evaluation games between trained model and baseline"""
        logger.info(f"Running {num_games} evaluation games")

        trained_wins = 0
        baseline_wins = 0
        draws = 0
        total_moves = 0
        total_time = 0

        for game_idx in range(num_games):
            logger.debug(f"Running evaluation game {game_idx + 1}/{num_games}")

            # Create new MCTS instance for evaluation
            eval_mcts_config = MCTSConfig(
                num_simulations=self.config['grpo'].get('mcts_simulations', 25),  # Reduced for faster eval
                cpuct=self.config['grpo'].get('cpuct', 2.2),
                virtual_loss=self.config['grpo'].get('virtual_loss', 2.0),
                max_children=self.config['grpo'].get('max_children', 64),
                min_child_prior=self.config['grpo'].get('min_child_prior', 0.0001),
                playout_random_frac=self.config['grpo'].get('playout_random_frac', 0.0),
                enable_tt_cache=self.config['grpo'].get('enable_tt_cache', False),
                tt_capacity=self.config['grpo'].get('tt_capacity', 100000)
            )

            # Load baseline model for comparison
            baseline_model = self._load_baseline_model()
            trained_mcts = MCTS(self.model, eval_mcts_config, self.device)
            baseline_mcts = MCTS(baseline_model, eval_mcts_config, self.device)

            # Play game between trained and baseline
            game_start = time.time()
            result, moves = self._play_evaluation_game(trained_mcts, baseline_mcts)
            game_time = time.time() - game_start

            total_moves += moves
            total_time += game_time

            if result == 1:
                trained_wins += 1
            elif result == -1:
                baseline_wins += 1
            else:
                draws += 1

            logger.debug(f"Game {game_idx + 1}: {'Trained' if result == 1 else 'Baseline' if result == -1 else 'Draw'} ({moves} moves, {game_time:.2f}s)")

        # Calculate metrics
        total_games = trained_wins + baseline_wins + draws
        win_rate = trained_wins / total_games if total_games > 0 else 0
        draw_rate = draws / total_games if total_games > 0 else 0
        avg_game_length = total_moves / total_games if total_games > 0 else 0
        avg_ms_per_move = (total_time * 1000) / total_moves if total_moves > 0 else 0

        return {
            'epoch': self.epoch,
            'games_played': total_games,
            'trained_wins': trained_wins,
            'baseline_wins': baseline_wins,
            'draws': draws,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'avg_game_length': avg_game_length,
            'avg_ms_per_move': avg_ms_per_move
        }

    def _create_dummy_trajectories(self) -> List[Any]:
        """Create dummy trajectories for testing that match Trajectory dataclass"""
        # Import here to avoid circular imports
        sys.path.append(str(Path(__file__).parent.parent))
        from training.grpo_trainer import Trajectory, TrajectoryStep

        trajectories = []
        for i in range(4):  # Group size
            # Create trajectory steps
            steps = []
            total_reward = 0.0

            for j in range(10):  # 10 steps
                step = TrajectoryStep(
                    state=torch.randn(19, 8, 8),  # Dummy board state
                    action=j % 4672,  # Dummy action
                    log_prob=-1.0 + np.random.normal(0, 0.1),  # Dummy log prob
                    value=np.random.normal(0, 0.5),  # Dummy value
                    reward=np.random.normal(0, 0.1),  # Dummy reward
                    done=(j == 9),  # Last step is done
                    legal_mask=torch.ones(4672) if j < 9 else None
                )
                steps.append(step)
                total_reward += step.reward

            # Create trajectory
            trajectory = Trajectory(
                steps=steps,
                total_reward=total_reward,
                length=len(steps),
                game_result=np.random.choice([-1.0, 0.0, 1.0])  # Random game result
            )
            trajectories.append(trajectory)

        return trajectories

    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path("results/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"grpo_checkpoint_epoch_{self.epoch}.pt"
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.grpo_trainer.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _create_display_content(self) -> str:
        """Create the display content as a single string"""
        uptime = datetime.now() - self.start_time
        last_heartbeat = datetime.now() - self.last_heartbeat

        perf = self.metrics['performance']

        # Calculate current epoch stats
        total_moves = sum(len(traj) for traj in self.current_trajectories)
        avg_ms_per_move = perf.get('avg_ms_per_move', 0.0)
        current_phase = getattr(self, 'current_phase', 'Initializing')

        content = f"""
🎯 GRPO Chess Experiment - Epoch {self.epoch}
Started: {self.start_time.strftime('%H:%M:%S')} | Games: {self.total_games} | Moves: {total_moves} | Uptime: {str(uptime).split('.')[0]}

📊 Performance Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Metric               ┃ Value   ┃ Target  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ Win Rate             │ {perf['win_rate']:.3f}   │ ≥ 0.30  │
│ Draw Rate            │ {perf['draw_rate']:.3f}   │ ≤ 0.30  │
│ Avg Game Length      │ {perf['avg_game_length']:.1f}    │ 60-90   │
│ Avg ms/move          │ {avg_ms_per_move:.1f}    │ ≤ 200   │
│ Games/sec            │ {perf['games_per_second']:.2f}    │ ≥ 1.0   │
└───────────────────────┴─────────┴─────────┘

🔄 Status: 💓 {last_heartbeat.seconds}s ago | 🎮 {self.device} | 🧠 {self.config['model']['type']} | ⚡ {self.config['grpo']['mcts_simulations']} sims
📍 Phase: {current_phase}
        """.strip()

        return content

    def _create_performance_table(self) -> Table:
        """Create performance metrics table"""
        table = Table(title="📊 Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Target", style="green")

        perf = self.metrics['performance']
        table.add_row("Win Rate", f"{perf['win_rate']:.3f}", "≥ 0.30")
        table.add_row("Draw Rate", f"{perf['draw_rate']:.3f}", "≤ 0.30")
        table.add_row("Avg Game Length", f"{perf['avg_game_length']:.1f}", "60-90")
        table.add_row("Games/sec", f"{perf['games_per_second']:.2f}", "≥ 1.0")

        return table

    def _create_status_panel(self) -> Panel:
        """Create current status panel"""
        uptime = datetime.now() - self.start_time
        last_heartbeat = datetime.now() - self.last_heartbeat

        status_text = (
            f"⏱️  Uptime: {str(uptime).split('.')[0]} | "
            f"💓 Heartbeat: {last_heartbeat.seconds}s ago | "
            f"🎮 Device: {self.device} | "
            f"🧠 Model: {self.config['model']['type']} | "
            f"⚡ MCTS Sims: {self.config['grpo']['mcts_simulations']}"
        )

        return Panel(status_text, title="🔄 Current Status", border_style="green")

    def _start_heartbeat(self):
        """Start heartbeat monitoring thread"""
        def heartbeat():
            while True:
                self.last_heartbeat = datetime.now()
                # Update display periodically
                if hasattr(self, 'live'):
                    try:
                        self.live.update(self._create_display_content())
                    except Exception:
                        pass  # Ignore display update errors
                time.sleep(self.heartbeat_interval)

        self.display_thread = threading.Thread(target=heartbeat, daemon=True)
        self.display_thread.start()

    def _stop_heartbeat(self):
        """Stop heartbeat monitoring"""
        if self.display_thread:
            self.display_thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description='GRPO Chess Orchestrator')
    parser.add_argument('--config', type=str, default='medium_transformer_grpo',
                       help='Experiment configuration name')
    parser.add_argument('--games', type=int, default=30,
                       help='Number of self-play games per epoch')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda/mps)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/baseline_medium_transformer.pt',
                       help='Path to model checkpoint')

    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'experiment_configs.yaml'
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if args.config not in configs:
        print(f"❌ Configuration '{args.config}' not found")
        sys.exit(1)

    config = configs[args.config]
    config['checkpoint_path'] = args.checkpoint

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"🚀 Starting GRPO Orchestrator")
    print(f"   Config: {args.config}")
    print(f"   Games: {args.games}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Device: {args.device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print()

    # Create and run orchestrator
    orchestrator = GRPOOrchestrator(config, args.device)
    orchestrator.run_experiment(args.games, args.epochs)


if __name__ == "__main__":
    import yaml
    main()
