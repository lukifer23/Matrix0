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
import chess.pgn

import torch

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig
from ..encoding import encode_board, move_to_index
from ..engines import EngineManager
import chess.polyglot as polyglot
import io
import os

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
        sp_cfg = config.selfplay()
        self.matrix0_mcts = MCTS(
            self.matrix0_model,
            MCTSConfig(
                num_simulations=int(sp_cfg.get("num_simulations", 200)),
                cpuct=float(sp_cfg.get("cpuct", 1.5)),
                dirichlet_alpha=float(sp_cfg.get("dirichlet_alpha", 0.3)),
                dirichlet_frac=float(sp_cfg.get("dirichlet_frac", 0.25)),
                tt_capacity=int(sp_cfg.get("tt_capacity", 200000)),
                selection_jitter=float(sp_cfg.get("selection_jitter", 0.0)),
            ),
            self.device,
        )
        
        # Load best checkpoint if available
        checkpoint_path = config.engines().get("matrix0", {}).get("checkpoint", "checkpoints/best.pt")
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            sd = state.get("model_ema") or state.get("model") or state
            try:
                missing, unexpected = self.matrix0_model.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    logger.warning(f"Checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path} (strict=False): {e}")
            else:
                logger.info(f"Loaded Matrix0 checkpoint: {checkpoint_path}")
        
        # Configuration
        self.external_engine_ratio = config.selfplay().get("external_engine_ratio", 0.3)
        self.engine_strength_curriculum = config.selfplay().get("engine_strength_curriculum", True)
        self.max_game_len = config.selfplay().get("max_game_len", 512)
        self.resign_threshold = config.selfplay().get("resign_threshold", -0.95)
        # Openings configuration
        self.openings_cfg = config.openings()
        self._book_path = self.openings_cfg.get("polyglot")
        self._pgn_path = self.openings_cfg.get("pgn")
        self._random_plies = int(self.openings_cfg.get("random_plies", 0))
        self._open_max_plies = int(self.openings_cfg.get("max_plies", 12))
        
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
        # Apply opening if configured
        try:
            self._apply_opening(board)
        except Exception as e:
            logger.warning(f"Opening application failed: {e}")
        moves = []
        game_data = {
            "white_engine": white_engine,
            "black_engine": black_engine,
            "moves": [],
            "positions": [],
            "fens": [],
            "evaluations": [],
            "multipv": []  # list per position: [{move, score_cp}, ...]
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
            game_data["fens"].append(board.fen())
            game_data["moves"].append(move.uci())
            
            # Get evaluation if available
            evaluation = await self._get_position_evaluation(board, current_engine)
            if evaluation is not None:
                game_data["evaluations"].append(evaluation)
            # Get MultiPV (only for external engines)
            multipv_list = await self._get_multipv(board, current_engine)
            if multipv_list is not None:
                game_data["multipv"].append(multipv_list)
            else:
                game_data["multipv"].append([])
            
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

    async def _get_multipv(self, board: chess.Board, engine_name: str):
        if engine_name == "matrix0":
            return None
        engine_client = self.engine_manager.get_engine(engine_name)
        if engine_client is None:
            return None
        time_control = self.config.engines().get(engine_name, {}).get("time_control", "100ms")
        time_ms = self._parse_time_control(time_control)
        multipv = int(self.config.external_data().get("multipv", 8))
        return await engine_client.analyze_multipv(board, time_ms=time_ms, multipv=multipv)
    
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

    def _apply_opening(self, board: chess.Board) -> None:
        # Try polyglot book
        if self._book_path and os.path.exists(self._book_path):
            try:
                with polyglot.open_reader(self._book_path) as reader:
                    plies = 0
                    while plies < self._open_max_plies:
                        entries = list(reader.find_all(board))
                        if not entries:
                            break
                        mv = random.choice(entries).move()
                        if mv not in board.legal_moves:
                            break
                        board.push(mv)
                        plies += 1
                return
            except Exception as e:
                logger.warning(f"Polyglot opening failed: {e}")
        # Try PGN
        if self._pgn_path and os.path.exists(self._pgn_path):
            try:
                with open(self._pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read first game only for simplicity; in production sample randomly
                    game = chess.pgn.read_game(f)
                    if game is not None:
                        node = game
                        plies = 0
                        while node.variations and plies < self._open_max_plies:
                            node = random.choice(node.variations)
                            board.push(node.move)
                            plies += 1
                        return
            except Exception as e:
                logger.warning(f"PGN opening failed: {e}")
        # Random plies as last resort
        rp = max(0, int(self._random_plies))
        for _ in range(rp):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))


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
