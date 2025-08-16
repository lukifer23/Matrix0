"""Multi-engine evaluator for Matrix0 against external engines."""

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from time import perf_counter

import chess
import torch

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..engines import EngineManager

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an evaluation match."""
    matrix0_wins: int
    matrix0_losses: int
    matrix0_draws: int
    total_games: int
    win_rate: float
    engine_name: str
    time_control: str
    games: List[Dict[str, Any]]


class MultiEngineEvaluator:
    """Evaluates Matrix0 against multiple external engines."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = select_device(config.get("device", "auto"))
        
        # Load Matrix0 model
        self.matrix0_model = PolicyValueNet.from_config(config.model()).to(self.device)
        self.matrix0_mcts = MCTS(self.matrix0_model, MCTSConfig(**config.eval()), self.device)
        
        # Load best checkpoint if available
        checkpoint_path = config.engines().get("matrix0", {}).get("checkpoint", "checkpoints/best.pt")
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            if "model_ema" in state:
                self.matrix0_model.load_state_dict(state["model_ema"])
            else:
                self.matrix0_model.load_state_dict(state["model"])
            logger.info(f"Loaded Matrix0 checkpoint: {checkpoint_path}")
        
        # Configuration
        self.eval_config = config.eval()
        self.engine_configs = config.engines()
        
    async def evaluate_against_engines(self, engine_names: Optional[List[str]] = None, 
                                    games_per_engine: int = 50) -> Dict[str, EvaluationResult]:
        """Evaluate Matrix0 against specified engines."""
        if engine_names is None:
            engine_names = self.eval_config.get("external_engines", ["stockfish", "lc0"])
        
        # Filter to enabled engines only
        available_engines = [
            name for name in engine_names 
            if name in self.engine_configs and self.engine_configs[name].get("enabled", False)
        ]
        
        if not available_engines:
            logger.warning("No external engines available for evaluation")
            return {}
        
        logger.info(f"Evaluating Matrix0 against engines: {available_engines}")
        
        # Initialize engine manager
        engine_manager = EngineManager(self.config.to_dict())
        await engine_manager.start_all_engines()
        
        results = {}
        
        try:
            for engine_name in available_engines:
                logger.info(f"Evaluating against {engine_name}")
                
                result = await self._evaluate_against_engine(
                    engine_manager, engine_name, games_per_engine
                )
                
                if result:
                    results[engine_name] = result
                    logger.info(f"Evaluation against {engine_name}: {result.win_rate:.3f} win rate")
                
        finally:
            await engine_manager.cleanup()
        
        return results
    
    async def _evaluate_against_engine(self, engine_manager: EngineManager, 
                                     engine_name: str, games: int) -> Optional[EvaluationResult]:
        """Evaluate Matrix0 against a specific engine."""
        if not engine_manager.is_engine_healthy(engine_name):
            logger.error(f"Engine {engine_name} is not healthy")
            return None
        
        matrix0_wins = 0
        matrix0_losses = 0
        matrix0_draws = 0
        game_results = []
        
        for game_num in range(games):
            logger.debug(f"Playing game {game_num + 1}/{games} against {engine_name}")
            
            # Alternate colors
            matrix0_plays_white = (game_num % 2 == 0)
            
            try:
                game_result = await self._play_single_game(
                    engine_manager, engine_name, matrix0_plays_white
                )
                
                if game_result:
                    # Determine result from Matrix0's perspective
                    if matrix0_plays_white:
                        if game_result["result"] == 1.0:
                            matrix0_wins += 1
                        elif game_result["result"] == -1.0:
                            matrix0_losses += 1
                        else:
                            matrix0_draws += 1
                    else:
                        if game_result["result"] == -1.0:
                            matrix0_wins += 1
                        elif game_result["result"] == 1.0:
                            matrix0_losses += 1
                        else:
                            matrix0_draws += 1
                    
                    game_results.append(game_result)
                    
            except Exception as e:
                logger.error(f"Game {game_num + 1} failed: {e}")
                continue
        
        total_games = matrix0_wins + matrix0_losses + matrix0_draws
        if total_games == 0:
            logger.warning(f"No valid games completed against {engine_name}")
            return None
        
        win_rate = matrix0_wins / total_games
        
        # Get time control from config
        time_control = self.engine_configs.get(engine_name, {}).get("time_control", "100ms")
        
        return EvaluationResult(
            matrix0_wins=matrix0_wins,
            matrix0_losses=matrix0_losses,
            matrix0_draws=matrix0_draws,
            total_games=total_games,
            win_rate=win_rate,
            engine_name=engine_name,
            time_control=time_control,
            games=game_results
        )
    
    async def _play_single_game(self, engine_manager: EngineManager, 
                               engine_name: str, matrix0_plays_white: bool) -> Optional[Dict[str, Any]]:
        """Play a single game between Matrix0 and an external engine."""
        board = chess.Board()
        moves = []
        game_data = {
            "moves": [],
            "positions": [],
            "matrix0_evaluations": [],
            "external_evaluations": []
        }
        
        start_time = perf_counter()
        
        while not board.is_game_over() and len(moves) < 512:  # Max game length
            current_engine = "matrix0" if (
                (matrix0_plays_white and board.turn == chess.WHITE) or
                (not matrix0_plays_white and board.turn == chess.BLACK)
            ) else engine_name
            
            # Get move from appropriate engine
            move = await self._get_engine_move(board, engine_manager, current_engine)
            if move is None:
                logger.error(f"Engine {current_engine} failed to provide move")
                return None
            
            # Validate move
            if move not in board.legal_moves:
                logger.error(f"Engine {current_engine} provided illegal move: {move}")
                return None
            
            # Record position and move
            game_data["moves"].append(move.uci())
            
            # Get evaluations from both engines
            matrix0_eval = await self._get_matrix0_evaluation(board)
            external_eval = await self._get_external_evaluation(board, engine_manager, engine_name)
            
            if matrix0_eval is not None:
                game_data["matrix0_evaluations"].append(matrix0_eval)
            if external_eval is not None:
                game_data["external_evaluations"].append(external_eval)
            
            # Make move
            board.push(move)
            moves.append(move)
        
        # Determine game result
        if board.is_game_over():
            result = self._get_game_result(board)
        else:
            # Game truncated
            result = 0.0  # Draw
        
        game_time = perf_counter() - start_time
        
        return {
            "result": result,
            "moves": len(moves),
            "time_seconds": game_time,
            "final_fen": board.fen(),
            "game_data": game_data
        }
    
    async def _get_engine_move(self, board: chess.Board, engine_manager: EngineManager, 
                              engine_name: str) -> Optional[chess.Move]:
        """Get a move from the specified engine."""
        if engine_name == "matrix0":
            # Use Matrix0 MCTS
            visit_counts, pi, v = self.matrix0_mcts.run(board)
            move = max(visit_counts.items(), key=lambda kv: kv[1])[0]
            return move
        else:
            # Use external engine
            engine_client = engine_manager.get_engine(engine_name)
            if engine_client is None:
                return None
            
            # Get time control from config
            time_control = self.engine_configs.get(engine_name, {}).get("time_control", "100ms")
            time_ms = self._parse_time_control(time_control)
            
            move = await engine_client.get_move(board, time_ms)
            return move
    
    async def _get_matrix0_evaluation(self, board: chess.Board) -> Optional[float]:
        """Get Matrix0's evaluation of a position."""
        try:
            from ..encoding import encode_board
            with torch.no_grad():
                encoded_board = encode_board(board)
                encoded_tensor = torch.from_numpy(encoded_board).unsqueeze(0).to(self.device)
                policy, value = self.matrix0_model(encoded_tensor)
                return value.item()
        except Exception as e:
            logger.debug(f"Failed to get Matrix0 evaluation: {e}")
            return None
    
    async def _get_external_evaluation(self, board: chess.Board, engine_manager: EngineManager, 
                                     engine_name: str) -> Optional[float]:
        """Get external engine's evaluation of a position."""
        try:
            engine_client = engine_manager.get_engine(engine_name)
            if engine_client is None:
                return None
            
            analysis = await engine_client.analyze_position(board, depth=1)
            if analysis and analysis.get("score"):
                score = analysis["score"]
                # Convert chess.engine.Score to float
                if hasattr(score, 'white'):
                    return score.white().score(mate_score=10000) / 100.0
                elif hasattr(score, 'score'):
                    return score.score(mate_score=10000) / 100.0
            return None
        except Exception as e:
            logger.debug(f"Failed to get external engine evaluation: {e}")
            return None
    
    def _get_game_result(self, board: chess.Board) -> float:
        """Get game result as float."""
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
    
    def _parse_time_control(self, time_control: str) -> int:
        """Parse time control string to milliseconds."""
        try:
            if time_control.endswith("ms"):
                return int(time_control[:-2])
            elif time_control.endswith("s"):
                return int(float(time_control[:-1]) * 1000)
            else:
                return int(float(time_control) * 1000)
        except (ValueError, AttributeError):
            return 100  # Default to 100ms


async def evaluate_matrix0_against_engines(config_path: str, engine_names: Optional[List[str]] = None, 
                                         games_per_engine: int = 50) -> Dict[str, EvaluationResult]:
    """Convenience function to evaluate Matrix0 against external engines."""
    config = Config.load(config_path)
    evaluator = MultiEngineEvaluator(config)
    return await evaluator.evaluate_against_engines(engine_names, games_per_engine)


def main():
    """Command-line interface for multi-engine evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Matrix0 against external engines")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--engines", nargs="+", help="Engines to evaluate against")
    parser.add_argument("--games", type=int, default=50, help="Games per engine")
    
    args = parser.parse_args()
    
    async def run_evaluation():
        results = await evaluate_matrix0_against_engines(
            args.config, args.engines, args.games
        )
        
        print("\nEvaluation Results:")
        print("=" * 50)
        
        for engine_name, result in results.items():
            print(f"\n{engine_name.upper()}:")
            print(f"  Games: {result.total_games}")
            print(f"  Wins: {result.matrix0_wins}")
            print(f"  Losses: {result.matrix0_losses}")
            print(f"  Draws: {result.matrix0_draws}")
            print(f"  Win Rate: {result.win_rate:.3f}")
            print(f"  Time Control: {result.time_control}")
    
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
