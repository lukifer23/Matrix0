#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script

Features:
- Fast evaluation with different simulation levels
- TUI table with live updates
- Game saving (PGN) for analysis
- ELO rating updates
- Variance analysis between models
- Temperature control for different playing styles
- Comprehensive logging for debugging
"""

import argparse
import chess
import chess.pgn
import torch
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..elo import EloBook, update_elo
from ..draw import should_adjudicate_draw


class EnhancedEvaluator:
    def __init__(self, config_path: str, model_a_path: str, model_b_path: str):
        self.console = Console()
        
        # Setup logging
        self._setup_logging()
        
        self.cfg = Config.load(config_path)
        self.device = select_device(self.cfg.get("device", "auto"))
        
        # Model paths
        self.model_a_path = Path(model_a_path)
        self.model_b_path = Path(model_b_path)
        
        # ELO tracking - initialize with default values if file doesn't exist
        self.elo_book = EloBook(Path("data/elo_ratings.json"))
        try:
            self.elo_data = self.elo_book.load()
            self.logger.info(f"Loaded ELO data: {self.elo_data}")
        except Exception as e:
            self.logger.warning(f"Could not load ELO data, using defaults: {e}")
            self.elo_data = {"enhanced_best": 1500.0, "best": 1500.0}
        
        # Results tracking
        self.results = []
        self.current_game = 0
        self.total_games = 9  # 3 games × 3 simulation levels
        
        # Create output directory
        self.output_dir = Path("data/eval_games")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self._load_models()
        
    def _setup_logging(self):
        """Setup comprehensive logging for debugging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/eval_debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_models(self):
        """Load both models with proper PyTorch 2.6 compatibility."""
        self.console.print("🔄 Loading models...", style="blue")
        self.logger.info("Starting model loading process")
        
        try:
            # Create models
            self.logger.info("Creating model instances")
            self.model_a = PolicyValueNet.from_config(self.cfg.model()).to(self.device)
            self.model_b = PolicyValueNet.from_config(self.cfg.model()).to(self.device)
            self.logger.info("Models created successfully")
            
            # Load checkpoints with weights_only=False for PyTorch 2.6
            self.logger.info(f"Loading checkpoint A: {self.model_a_path}")
            checkpoint_a = torch.load(self.model_a_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Checkpoint A loaded, keys: {len(checkpoint_a.keys())}")
            
            self.logger.info(f"Loading checkpoint B: {self.model_b_path}")
            checkpoint_b = torch.load(self.model_b_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Checkpoint B loaded, keys: {len(checkpoint_b.keys())}")
            
            # Load state dicts
            self.logger.info("Loading state dict for model A")
            self.model_a.load_state_dict(checkpoint_a.get("model_ema", checkpoint_a.get("model", checkpoint_a)))
            self.logger.info("State dict loaded for model A")
            
            self.logger.info("Loading state dict for model B")
            self.model_b.load_state_dict(checkpoint_b.get("model_ema", checkpoint_b.get("model", checkpoint_b)))
            self.logger.info("State dict loaded for model B")
            
            self.model_a.eval()
            self.model_b.eval()
            self.logger.info("Models set to evaluation mode")
            
            self.console.print("✅ Models loaded successfully!", style="green")
            self.logger.info("Model loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}", exc_info=True)
            raise
        
    def _create_mcts_config(self, num_simulations: int, temperature: float) -> MCTSConfig:
        """Create unified MCTS config with specified parameters."""
        self.logger.info(f"Creating MCTS config: {num_simulations} sims, temp {temperature}")
        mcfg_dict = dict(self.cfg.mcts())
        mcfg_dict.update({
            "num_simulations": num_simulations,
            "selection_jitter": float(self.cfg.selfplay().get("selection_jitter", 0.0)),
            "batch_size": 1,
            "fpu": float(self.cfg.mcts().get("fpu", 0.5)),
            "parent_q_init": bool(self.cfg.mcts().get("parent_q_init", True)),
            "tt_cleanup_frequency": int(self.cfg.mcts().get("tt_cleanup_frequency", 500)),
            "draw_penalty": float(self.cfg.mcts().get("draw_penalty", -0.1)),
        })
        
        # Add aggressive settings to force decisive games
        if temperature > 1.0:  # Fast/Medium configs
            mcfg_dict.update({
                "cpuct": 1.5,  # Lower exploration for more decisive play
                "fpu": 0.3,    # Lower first play urgency
                "draw_penalty": -0.5,  # Stronger draw penalty
                "selection_jitter": 0.1,  # Add some randomness
            })
        else:  # Precise config
            mcfg_dict.update({
                "cpuct": 2.0,  # Balanced exploration
                "fpu": 0.4,
                "draw_penalty": -0.3,
                "selection_jitter": 0.05,
            })
        
        mcfg = MCTSConfig.from_dict(mcfg_dict)
        self.logger.info(f"MCTS config created: {mcfg}")
        return mcfg
    
    def _test_model_forward_pass(self, model, test_input, model_name: str):
        """Test model forward pass with detailed timing."""
        self.logger.info(f"Testing forward pass for {model_name}")
        
        try:
            model.eval()
            with torch.no_grad():
                self.logger.info(f"Starting forward pass for {model_name}")
                start_time = time.time()
                
                # Test forward pass
                policy, value = model(test_input)
                
                forward_time = time.time() - start_time
                self.logger.info(f"Forward pass completed for {model_name} in {forward_time:.3f}s")
                self.logger.info(f"Output shapes - Policy: {policy.shape}, Value: {value.shape}")
                
                return True, forward_time
                
        except Exception as e:
            self.logger.error(f"Forward pass failed for {model_name}: {e}", exc_info=True)
            return False, 0
    
    def _play_game(self, mcts_a: MCTS, mcts_b: MCTS, temperature: float, table: Table = None, game_num: int = None) -> Tuple[float, List[str], str, Dict]:
        """Play a complete game between two models."""
        self.logger.info(f"Starting new game with temperature {temperature}, max moves 200")
        
        # Test model forward passes before game start
        self.logger.info("Testing model forward passes before game start")
        test_input = torch.randn(1, 19, 8, 8, device=self.device)
        self._test_model_forward_pass(self.model_a, test_input, "Model A")
        self._test_model_forward_pass(self.model_b, test_input, "Model B")
        
        # Initialize game
        board = chess.Board()
        moves = []
        move_times = []
        move_count = 0
        start_time = time.time()
        
        # Track move choices for variance analysis
        move_choices = {"a": [], "b": []}
        
        # Use the passed MCTS instances directly (they are fresh for each game)
        self.logger.info(f"Using MCTS instances - A: {id(mcts_a)}, B: {id(mcts_b)}")
        
        # Add live progress row to table if provided
        if table and game_num:
            # Add a progress row that we can update
            table.add_row(
                str(game_num),
                "In Progress...",
                "Playing...",
                "0",
                "0.0s",
                "1500",
                "1500",
                "Calculating..."
            )
        
        while move_count < 200 and not board.is_game_over():
            move_start = time.time()
            
            # Determine which model's turn
            if board.turn == chess.WHITE:
                model_name = "A"
                model_key = "a"
                mcts = mcts_a
            else:
                model_name = "B"
                model_key = "b"
                mcts = mcts_b
            
            self.logger.info(f"Move {move_count + 1}: Using {model_name} (MCTS instance: {id(mcts)})")
            self.logger.info(f"Move {move_count + 1}: Board FEN: {board.fen()[:50]}...")
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            self.logger.info(f"Move {move_count + 1}: Legal moves: {len(legal_moves)}")
            
            # Run MCTS
            self.logger.info(f"Move {move_count + 1}: Starting MCTS run")
            visits, policy, vroot = mcts.run(board)
            self.logger.info(f"Move {move_count + 1}: MCTS completed in {time.time() - move_start:.3f}s")
            self.logger.info(f"Move {move_count + 1}: MCTS returned {len(visits)} visit counts")
            
            # Select move with temperature
            self.logger.info(f"Move {move_count + 1}: Selecting move with temperature {temperature}")
            if temperature > 1e-3:
                # Apply temperature to visit counts for sampling
                legal_moves = list(visits.keys())
                visit_counts = np.array([visits[m] for m in legal_moves], dtype=np.float32)
                
                # Apply temperature
                logits = np.log(visit_counts + 1e-8) / temperature
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                
                # Sample move
                move_idx = np.random.choice(len(legal_moves), p=probs)
                move = legal_moves[move_idx]
                self.logger.info(f"Move {move_count + 1}: Sampled move {move} (idx {move_idx})")
            else:
                # Greedy selection
                move = max(visits.keys(), key=visits.get)
                self.logger.info(f"Move {move_count + 1}: Greedy move {move}")
            
            # Record move choice for variance analysis
            move_choices[model_key].append({
                "move": str(move),
                "visits": visits.get(move, 0),
                "value": float(vroot),
                "legal_moves": len(visits),
                "move_number": move_count + 1,
                "model": model_name
            })
            
            self.logger.debug(f"Move {move_count + 1}: Recorded choice for {model_key} - move: {move}, visits: {visits.get(move, 0)}, value: {vroot:.3f}")
            
            # Make move
            self.logger.info(f"Move {move_count + 1}: Making move {move}")
            board.push(move)
            moves.append(str(move))
            
            # Record time
            move_time = time.time() - move_start
            move_times.append(move_time)
            self.logger.info(f"Move {move_count + 1}: Completed in {move_time:.3f}s")
            
            # Simple progress update every 10 moves (no complex table manipulation)
            if move_count % 10 == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(f"Game progress: {move_count} moves completed, elapsed time: {elapsed_time:.1f}s")
            
            # Check for draw adjudication
            if should_adjudicate_draw(board, moves, self.cfg.draw()):
                self.logger.info(f"Move {move_count + 1}: Draw adjudicated")
                break
            
            move_count += 1
            
        # Determine result
        self.logger.info("Game completed, determining result")
        if board.is_checkmate():
            if board.turn == chess.BLACK:  # White won
                result = 1.0
                result_str = "1-0"
            else:  # Black won
                result = 0.0
                result_str = "0-1"
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            result = 0.5
            result_str = "1/2-1/2"
        else:
            result = 0.5
            result_str = "1/2-1/2"
        
        game_time = time.time() - start_time
        self.logger.info(f"Game result: {result_str} in {move_count} moves, total time: {game_time:.2f}s")
        
        # Create game metadata
        game_meta = {
            "result": result,
            "result_str": result_str,
            "moves": moves,
            "move_times": move_times,
            "total_time": game_time,
            "move_choices": move_choices,
            "final_fen": board.fen(),
            "game_over_reason": self._get_game_over_reason(board)
        }
        
        return result, moves, result_str, game_meta
    
    def _get_game_over_reason(self, board: chess.Board) -> str:
        """Get the reason why the game ended."""
        if board.is_checkmate():
            return "checkmate"
        elif board.is_stalemate():
            return "stalemate"
        elif board.is_insufficient_material():
            return "insufficient_material"
        elif board.is_fifty_moves():
            return "fifty_moves"
        elif board.is_repetition():
            return "repetition"
        else:
            return "unknown"
    
    def _save_game_pgn(self, game_meta: Dict, game_num: int, sim_level: str) -> str:
        """Save game as PGN file."""
        self.logger.info(f"Saving game {game_num} ({sim_level}) as PGN")
        
        # Create PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = f"Matrix0 Enhanced Evaluation Game {game_num}"
        game.headers["Site"] = "Matrix0 Arena"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = f"{sim_level} - Game {game_num}"
        game.headers["White"] = "Model A (trained)"
        game.headers["Black"] = "Model B (baseline)"
        game.headers["Result"] = game_meta["result_str"]
        game.headers["TimeControl"] = "unlimited"
        game.headers["Termination"] = game_meta["game_over_reason"]
        
        # Replay the game on a fresh board to get correct SAN notation
        board = chess.Board()
        current_node = game
        
        for move_uci in game_meta["moves"]:
            try:
                # Parse the UCI move
                move = chess.Move.from_uci(move_uci)
                
                # Verify the move is legal in the current position
                if move in board.legal_moves:
                    # Make the move on the board
                    board.push(move)
                    
                    # Add the move to the PGN (this will automatically convert to SAN)
                    current_node = current_node.add_main_variation(move)
                else:
                    self.logger.warning(f"Invalid move {move_uci} at position {board.fen()[:50]}...")
                    # Skip invalid moves but continue
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error processing move {move_uci}: {e}")
                # Skip problematic moves but continue
                continue
        
        # Save PGN
        pgn_filename = f"eval_game_{sim_level}_{game_num:02d}_{game_meta['result_str'].replace('/', '-')}.pgn"
        pgn_path = self.output_dir / pgn_filename
        
        try:
            with open(pgn_path, 'w') as f:
                f.write(str(game))
            self.logger.info(f"Game saved to: {pgn_path}")
        except Exception as e:
            self.logger.error(f"Error writing PGN file: {e}")
            # Create a fallback text file with game data
            fallback_path = pgn_path.with_suffix('.txt')
            with open(fallback_path, 'w') as f:
                f.write(f"Game {game_num} ({sim_level})\n")
                f.write(f"Result: {game_meta['result_str']}\n")
                f.write(f"Moves: {' '.join(game_meta['moves'])}\n")
                f.write(f"Final FEN: {game_meta['final_fen']}\n")
            self.logger.info(f"Fallback game data saved to: {fallback_path}")
            return str(fallback_path)
        
        return str(pgn_path)
    
    def _calculate_variance(self, move_choices: Dict) -> Dict:
        """Calculate variance metrics between models."""
        self.logger.info(f"Calculating variance for move_choices: {move_choices}")
        
        if not move_choices["a"] or not move_choices["b"]:
            self.logger.warning(f"Empty move choices - A: {len(move_choices.get('a', []))}, B: {len(move_choices.get('b', []))}")
            return {}
        
        # Compare move selection patterns
        a_visits = [m["visits"] for m in move_choices["a"]]
        b_visits = [m["visits"] for m in move_choices["b"]]
        
        a_values = [m["value"] for m in move_choices["a"]]
        b_values = [m["value"] for m in move_choices["b"]]
        
        self.logger.info(f"Visit counts - A: {a_visits[:5]}..., B: {b_visits[:5]}...")
        self.logger.info(f"Value scores - A: {a_values[:5]}..., B: {b_values[:5]}...")
        
        # Calculate visit distribution variance
        if len(a_visits) > 1 and len(b_visits) > 1:
            visit_variance = np.var(a_visits + b_visits)
        else:
            visit_variance = 0
            
        # Calculate value variance
        if len(a_values) > 1 and len(b_values) > 1:
            value_variance = np.var(a_values + b_values)
        else:
            value_variance = 0
        
        # Calculate move diversity (how many different moves were chosen)
        a_moves = set(m["move"] for m in move_choices["a"])
        b_moves = set(m["move"] for m in move_choices["b"])
        move_diversity = len(a_moves.union(b_moves))
        
        # Calculate average metrics
        avg_visits_a = np.mean(a_visits) if a_visits else 0
        avg_visits_b = np.mean(b_visits) if b_visits else 0
        avg_value_a = np.mean(a_values) if a_values else 0
        avg_value_b = np.mean(b_values) if b_values else 0
        
        # Calculate playing style differences
        style_diff = abs(avg_visits_a - avg_visits_b) + abs(avg_value_a - avg_value_b)
        
        variance_metrics = {
            "visit_variance": visit_variance,
            "value_variance": value_variance,
            "move_diversity": move_diversity,
            "style_difference": style_diff,
            "avg_visits_a": avg_visits_a,
            "avg_visits_b": avg_visits_b,
            "avg_value_a": avg_value_a,
            "avg_value_b": avg_value_b,
        }
        
        self.logger.info(f"Calculated variance metrics: {variance_metrics}")
        return variance_metrics
    
    def _create_tui_table(self) -> Table:
        """Create the TUI table for live updates."""
        table = Table(title="Matrix0 Enhanced Evaluation", show_header=True, header_style="bold magenta")
        
        table.add_column("Game", style="cyan", width=8)
        table.add_column("Sim Level", style="blue", width=12)
        table.add_column("Result", style="green", width=10)
        table.add_column("Moves", style="yellow", width=8)
        table.add_column("Time", style="red", width=10)
        table.add_column("ELO A", style="cyan", width=8)
        table.add_column("ELO B", style="cyan", width=8)
        table.add_column("Variance", style="magenta", width=10)
        
        return table
    
    def _update_elo_ratings(self, result: float):
        """Update ELO ratings after each game."""
        self.logger.info(f"Updating ELO ratings for result: {result}")
        
        # Get current ratings
        elo_a = self.elo_data.get("enhanced_best", 1500.0)
        elo_b = self.elo_data.get("best", 1500.0)
        
        self.logger.info(f"Current ELO - A: {elo_a:.1f}, B: {elo_b:.1f}")
        
        # Update ratings
        try:
            new_elo_a, new_elo_b = update_elo(elo_a, elo_b, result)
            self.logger.info(f"ELO calculation successful - A: {elo_a:.1f} → {new_elo_a:.1f}, B: {elo_b:.1f} → {new_elo_b:.1f}")
        except Exception as e:
            self.logger.error(f"ELO calculation failed: {e}")
            # Use simple ELO update as fallback
            k_factor = 32
            expected_a = 1 / (1 + 10**((elo_b - elo_a) / 400))
            expected_b = 1 - expected_a
            
            new_elo_a = elo_a + k_factor * (result - expected_a)
            new_elo_b = elo_b + k_factor * ((1 - result) - expected_b)
            self.logger.info(f"Used fallback ELO calculation - A: {elo_a:.1f} → {new_elo_a:.1f}, B: {elo_b:.1f} → {new_elo_b:.1f}")
        
        # Store updated ratings
        self.elo_data["enhanced_best"] = new_elo_a
        self.elo_data["best"] = new_elo_b
        
        # Save to file
        try:
            self.elo_book.save(self.elo_data)
            self.logger.info(f"ELO data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save ELO data: {e}")
        
        return new_elo_a, new_elo_b
    
    def run_evaluation(self):
        """Run the complete enhanced evaluation with comprehensive logging."""
        self.console.print(Panel.fit("🚀 Matrix0 Enhanced Evaluation", style="bold blue"))
        self.logger.info("Starting enhanced evaluation")
        
        # Evaluation parameters - more aggressive settings for decisive games
        eval_configs = [
            {"name": "Fast", "sims": 30, "temp": 2.0, "description": "Very aggressive, creative play"},
            {"name": "Medium", "sims": 80, "temp": 1.2, "description": "Balanced with aggression"},
            {"name": "Precise", "sims": 150, "temp": 0.8, "description": "Precise but not too conservative"}
        ]
        
        # Create initial TUI table
        table = self._create_tui_table()
        self.console.print(table)
        
        # Run evaluation with simple table updates
        for config in eval_configs:
            self.console.print(f"\n🎯 Running {config['name']} evaluation: {config['sims']} sims, temp {config['temp']}")
            self.logger.info(f"Starting {config['name']} evaluation: {config['sims']} sims, temp {config['temp']}")
            
            # Create MCTS instances for this config
            self.logger.info(f"Creating MCTS instances for {config['name']} config")
            mcfg = self._create_mcts_config(config["sims"], config["temp"])
            # mcts_a = MCTS(self.model_a, mcfg, self.device) # This line is removed
            # mcts_b = MCTS(self.model_b, mcfg, self.device) # This line is removed
            self.logger.info(f"MCTS instances created for {config['name']} config")
            
            # Play 3 games with this configuration
            for game_num in range(1, 4):
                self.current_game += 1
                self.logger.info(f"Starting game {self.current_game} ({config['name']} config, game {game_num})")
                
                try:
                    # Create completely fresh MCTS instances for each game to avoid state corruption
                    self.logger.info(f"Game {self.current_game}: Creating fresh MCTS instances")
                    mcts_a_fresh = MCTS(self.model_a, mcfg, self.device)
                    mcts_b_fresh = MCTS(self.model_b, mcfg, self.device)
                    self.logger.info(f"Game {self.current_game}: Fresh MCTS instances created - A: {id(mcts_a_fresh)}, B: {id(mcts_b_fresh)}")
                    
                    # Play game
                    self.logger.info(f"Game {self.current_game}: Starting game play")
                    result, moves, result_str, game_meta = self._play_game(
                        mcts_a_fresh, mcts_b_fresh, config["temp"], table, self.current_game
                    )
                    self.logger.info(f"Game {self.current_game}: Game completed successfully")
                    
                    # Calculate variance
                    self.logger.info(f"Game {self.current_game}: Calculating variance")
                    variance_metrics = self._calculate_variance(game_meta["move_choices"])
                    variance_score = variance_metrics.get("style_difference", 0)
                    
                    # Update ELO ratings
                    self.logger.info(f"Game {self.current_game}: Updating ELO ratings")
                    elo_a, elo_b = self._update_elo_ratings(result)
                    
                    # Save game
                    self.logger.info(f"Game {self.current_game}: Saving game")
                    pgn_path = self._save_game_pgn(game_meta, game_num, config["name"])
                    
                    # Store results
                    game_result = {
                        "game_num": self.current_game,
                        "config": config["name"],
                        "sims": config["sims"],
                        "temperature": config["temp"],
                        "result": result,
                        "result_str": result_str,
                        "moves": len(moves),
                        "time": game_meta["total_time"],
                        "elo_a": elo_a,
                        "elo_b": elo_b,
                        "variance": variance_score,
                        "pgn_path": pgn_path,
                        "variance_metrics": variance_metrics
                    }
                    self.results.append(game_result)
                    
                    # Add row to table and display updated table
                    table.add_row(
                        str(self.current_game),
                        config["name"],
                        result_str,
                        str(len(moves)),
                        f"{game_meta['total_time']:.1f}s",
                        f"{elo_a:.0f}",
                        f"{elo_b:.0f}",
                        f"{variance_score:.2f}"
                    )
                    
                    # Clear console and redisplay updated table
                    self.console.clear()
                    self.console.print(Panel.fit("🚀 Matrix0 Enhanced Evaluation", style="bold blue"))
                    self.console.print(table)
                    
                    # Progress update (log only, no console print to avoid TUI interference)
                    self.logger.info(f"Game {self.current_game} completed: {result_str} in {len(moves)} moves")
                    
                except Exception as e:
                    self.logger.error(f"Game {self.current_game} failed: {e}", exc_info=True)
                    self.console.print(f"❌ Game {self.current_game} failed: {e}", style="red")
                    raise
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print comprehensive evaluation summary."""
        self.console.print("\n" + "="*80)
        self.console.print("📊 EVALUATION SUMMARY", style="bold green")
        self.console.print("="*80)
        
        # Overall results
        total_games = len(self.results)
        wins_a = sum(1 for r in self.results if r["result"] == 1.0)
        wins_b = sum(1 for r in self.results if r["result"] == 0.0)
        draws = sum(1 for r in self.results if r["result"] == 0.5)
        
        self.console.print(f"🎮 Total Games: {total_games}")
        self.console.print(f"🏆 Model A Wins: {wins_a}")
        self.console.print(f"🏆 Model B Wins: {wins_b}")
        self.console.print(f"🤝 Draws: {draws}")
        self.console.print(f"📈 Win Rate A: {wins_a/total_games*100:.1f}%")
        self.console.print(f"📈 Win Rate B: {wins_b/total_games*100:.1f}%")
        
        # ELO changes
        initial_elo_a = 1500.0
        initial_elo_b = 1500.0
        final_elo_a = self.elo_data.get("enhanced_best", 1500.0)
        final_elo_b = self.elo_data.get("best", 1500.0)
        
        self.console.print(f"\n📊 ELO Changes:")
        self.console.print(f"  Model A: {initial_elo_a:.0f} → {final_elo_a:.0f} ({final_elo_a-initial_elo_a:+.0f})")
        self.console.print(f"  Model B: {initial_elo_b:.0f} → {final_elo_b:.0f} ({final_elo_b-initial_elo_b:+.0f})")
        
        # Performance by configuration
        self.console.print(f"\n⚙️  Performance by Configuration:")
        for config_name in ["Fast", "Medium", "Precise"]:
            config_results = [r for r in self.results if r["config"] == config_name]
            if config_results:
                wins_a_config = sum(1 for r in config_results if r["result"] == 1.0)
                total_config = len(config_results)
                self.console.print(f"  {config_name}: {wins_a_config}/{total_config} wins ({wins_a_config/total_config*100:.1f}%)")
        
        # Enhanced variance analysis
        self.console.print(f"\n🔍 Enhanced Variance Analysis:")
        if self.results:
            # Calculate aggregate variance metrics
            all_variance_metrics = [r.get("variance_metrics", {}) for r in self.results if r.get("variance_metrics")]
            if all_variance_metrics:
                avg_style_diff = np.mean([m.get("style_difference", 0) for m in all_variance_metrics])
                avg_move_diversity = np.mean([m.get("move_diversity", 0) for m in all_variance_metrics])
                avg_visit_variance = np.mean([m.get("visit_variance", 0) for m in all_variance_metrics])
                
                self.console.print(f"  Average Style Difference: {avg_style_diff:.2f}")
                self.console.print(f"  Average Move Diversity: {avg_move_diversity:.1f}")
                self.console.print(f"  Average Visit Variance: {avg_visit_variance:.1f}")
                
                # Interpret the results
                if avg_style_diff > 10:
                    self.console.print(f"  📊 Models show significant playing style differences")
                elif avg_style_diff > 5:
                    self.console.print(f"  📊 Models show moderate playing style differences")
                else:
                    self.console.print(f"  📊 Models show similar playing styles")
        
        # Game files
        self.console.print(f"\n💾 Game Files Saved:")
        for result in self.results:
            self.console.print(f"  {result['pgn_path']}")
        
        self.console.print("\n✅ Enhanced evaluation completed!")
        self.logger.info("Enhanced evaluation completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Matrix0 Model Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--model-a", required=True, help="Path to Model A (trained)")
    parser.add_argument("--model-b", required=True, help="Path to Model B (baseline)")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = EnhancedEvaluator(args.config, args.model_a, args.model_b)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
