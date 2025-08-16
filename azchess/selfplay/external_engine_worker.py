"""External engine self-play worker for generating training data."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from time import perf_counter

import numpy as np
import chess

import torch

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..encoding import encode_board, move_to_index
from ..engines import EngineManager

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a game between engines."""
    moves: int
    result: float  # 1.0 for white win, -1.0 for black win, 0.0 for draw
    time_seconds: float
    white_engine: str
    black_engine: str
    game_data: Dict[str, Any]


class ExternalEngineSelfPlay:
    """Generates games between Matrix0 and external engines."""
    
    def __init__(self, config: Config, engine_manager: EngineManager):
        self.config = config
        self.engine_manager = engine_manager
        self.device = select_device(config.get("device", "auto"))
        
        # Load Matrix0 model
        self.matrix0_model = PolicyValueNet.from_config(config.model()).to(self.device)
        self.matrix0_mcts = MCTS(self.matrix0_model, MCTSConfig(**config.selfplay()), self.device)
        
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
        self.external_engine_ratio = config.selfplay().get("external_engine_ratio", 0.3)
        self.engine_strength_curriculum = config.selfplay().get("engine_strength_curriculum", True)
        self.max_game_len = config.selfplay().get("max_game_len", 512)
        self.resign_threshold = config.selfplay().get("resign_threshold", -0.95)
        
    async def generate_games(self, num_games: int, output_dir: str, 
                           engine_pairs: Optional[List[Tuple[str, str]]] = None) -> List[GameResult]:
        """Generate games between different engine combinations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if engine_pairs is None:
            engine_pairs = self._generate_engine_pairs(num_games)
        
        games = []
        for i, (white_engine, black_engine) in enumerate(engine_pairs):
            logger.info(f"Generating game {i+1}/{num_games}: {white_engine} vs {black_engine}")
            
            try:
                game_result = await self._play_game(white_engine, black_engine)
                if game_result:
                    games.append(game_result)
                    
                    # Save game data
                    await self._save_game_data(game_result, output_path, i)
                    
            except Exception as e:
                logger.error(f"Failed to generate game {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(games)} games successfully")
        return games
    
    def _generate_engine_pairs(self, num_games: int) -> List[Tuple[str, str]]:
        """Generate engine pairs for games based on configuration."""
        pairs = []
        
        # Calculate how many games should be against external engines
        external_games = int(num_games * self.external_engine_ratio)
        internal_games = num_games - external_games
        
        # Internal Matrix0 vs Matrix0 games
        for _ in range(internal_games):
            pairs.append(("matrix0", "matrix0"))
        
        # External engine games
        available_external_engines = [
            name for name, config in self.config.engines().items()
            if config.get("enabled", False) and config.get("type") != "internal"
        ]
        
        if not available_external_engines:
            logger.warning("No external engines available, using internal games only")
            for _ in range(external_games):
                pairs.append(("matrix0", "matrix0"))
        else:
            for _ in range(external_games):
                # Randomly select external engines
                white_engine = random.choice(available_external_engines)
                black_engine = random.choice(available_external_engines)
                pairs.append((white_engine, black_engine))
        
        # Shuffle pairs for variety
        random.shuffle(pairs)
        return pairs
    
    async def _play_game(self, white_engine: str, black_engine: str) -> Optional[GameResult]:
        """Play a single game between two engines."""
        board = chess.Board()
        moves = []
        game_data = {
            "white_engine": white_engine,
            "black_engine": black_engine,
            "moves": [],
            "positions": [],
            "evaluations": []
        }
        
        start_time = perf_counter()
        
        while not board.is_game_over() and len(moves) < self.max_game_len:
            current_engine = white_engine if board.turn == chess.WHITE else black_engine
            
            # Get move from appropriate engine
            move = await self._get_engine_move(board, current_engine)
            if move is None:
                logger.error(f"Engine {current_engine} failed to provide move")
                return None
            
            # Validate move
            if move not in board.legal_moves:
                logger.error(f"Engine {current_engine} provided illegal move: {move}")
                return None
            
            # Record position and move
            game_data["positions"].append(encode_board(board))
            game_data["moves"].append(move.uci())
            
            # Get evaluation if available
            evaluation = await self._get_position_evaluation(board, current_engine)
            if evaluation is not None:
                game_data["evaluations"].append(evaluation)
            
            # Check for resignation
            if evaluation is not None and evaluation < self.resign_threshold:
                # Black resigns
                game_data["result"] = "resignation"
                game_data["resigning_engine"] = black_engine
                break
            
            # Make move
            board.push(move)
            moves.append(move)
        
        # Determine game result
        if board.is_game_over():
            result = self._get_game_result(board)
            game_data["result"] = "normal"
        else:
            # Game truncated or resigned
            result = game_data.get("result", "truncated")
            if result == "resignation":
                result = 1.0  # White wins by resignation
        
        game_data["final_fen"] = board.fen()
        game_data["move_count"] = len(moves)
        
        game_time = perf_counter() - start_time
        
        return GameResult(
            moves=len(moves),
            result=result,
            time_seconds=game_time,
            white_engine=white_engine,
            black_engine=black_engine,
            game_data=game_data
        )
    
    async def _get_engine_move(self, board: chess.Board, engine_name: str) -> Optional[chess.Move]:
        """Get a move from the specified engine."""
        if engine_name == "matrix0":
            # Use Matrix0 MCTS
            visit_counts, pi, v = self.matrix0_mcts.run(board)
            move = max(visit_counts.items(), key=lambda kv: kv[1])[0]
            return move
        else:
            # Use external engine
            engine_client = self.engine_manager.get_engine(engine_name)
            if engine_client is None:
                logger.error(f"Engine {engine_name} not available")
                return None
            
            # Get time control from config
            time_control = self.config.engines().get(engine_name, {}).get("time_control", "100ms")
            time_ms = self._parse_time_control(time_control)
            
            move = await engine_client.get_move(board, time_ms)
            return move
    
    async def _get_position_evaluation(self, board: chess.Board, engine_name: str) -> Optional[float]:
        """Get position evaluation from the specified engine."""
        if engine_name == "matrix0":
            # Use Matrix0 model evaluation
            with torch.no_grad():
                encoded_board = encode_board(board)
                encoded_tensor = torch.from_numpy(encoded_board).unsqueeze(0).to(self.device)
                policy, value = self.matrix0_model(encoded_tensor)
                return value.item()
        else:
            # Use external engine analysis
            engine_client = self.engine_manager.get_engine(engine_name)
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
    
    async def _save_game_data(self, game_result: GameResult, output_dir: Path, game_index: int):
        """Save game data to file."""
        filename = output_dir / f"external_game_{game_index:04d}.json"
        
        # Prepare data for saving
        save_data = {
            "metadata": {
                "white_engine": game_result.white_engine,
                "black_engine": game_result.black_engine,
                "moves": game_result.moves,
                "result": game_result.result,
                "time_seconds": game_result.time_seconds,
                "timestamp": perf_counter()
            },
            "game_data": game_result.game_data
        }
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.debug(f"Saved game data to {filename}")


async def external_engine_worker(proc_id: int, config: Config, output_dir: str, 
                               num_games: int, engine_pairs: Optional[List[Tuple[str, str]]] = None):
    """Worker function for external engine self-play."""
    logger.info(f"Starting external engine worker {proc_id}")
    
    try:
        # Initialize engine manager
        engine_manager = EngineManager(config.to_dict())
        await engine_manager.start_all_engines()
        
        # Initialize self-play
        selfplay = ExternalEngineSelfPlay(config, engine_manager)
        
        # Generate games
        games = await selfplay.generate_games(num_games, output_dir, engine_pairs)
        
        logger.info(f"Worker {proc_id} completed {len(games)} games")
        return games
        
    except Exception as e:
        logger.error(f"Worker {proc_id} failed: {e}")
        raise
    finally:
        # Cleanup
        if 'engine_manager' in locals():
            await engine_manager.cleanup()
