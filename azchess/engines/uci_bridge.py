"""UCI protocol bridge for external chess engines."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import chess
import chess.engine

logger = logging.getLogger(__name__)


class UCIClient:
    """UCI client for communicating with external chess engines."""
    
    def __init__(self, engine_path: str, parameters: Dict[str, Any], time_control: str = "100ms"):
        self.engine_path = Path(engine_path)
        self.parameters = parameters
        self.time_control = time_control
        self.process: Optional[chess.engine.SimpleEngine] = None
        self.is_ready = False
        
    async def start_engine(self) -> bool:
        """Start the engine and configure it."""
        try:
            if not self.engine_path.exists():
                logger.error(f"Engine not found: {self.engine_path}")
                return False
                
            # Start engine process
            self.process = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
            
            # Wait for engine to be ready
            await self.process.wait_until_ready()
            
            # Configure engine parameters
            for param, value in self.parameters.items():
                try:
                    await self.process.configure({param: value})
                except Exception as e:
                    logger.warning(f"Failed to set parameter {param}={value}: {e}")
            
            self.is_ready = True
            logger.info(f"Engine {self.engine_path.name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start engine {self.engine_path}: {e}")
            self.is_ready = False
            return False
    
    async def stop_engine(self):
        """Stop the engine process."""
        if self.process:
            try:
                await self.process.quit()
                self.process = None
                self.is_ready = False
                logger.info(f"Engine {self.engine_path.name} stopped")
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
    
    async def get_move(self, board: chess.Board, time_ms: int = 100) -> Optional[chess.Move]:
        """Get the best move from the engine for a given position."""
        if not self.is_ready or not self.process:
            logger.error("Engine not ready")
            return None
            
        try:
            # Convert time to chess.engine.Limit
            limit = chess.engine.Limit(time=time_ms / 1000.0)
            
            # Get best move
            result = await self.process.play(board, limit)
            
            if result.move:
                logger.debug(f"Engine move: {result.move} (time: {result.ponder_move})")
                return result.move
            else:
                logger.warning("Engine returned no move")
                return None
                
        except Exception as e:
            logger.error(f"Error getting move from engine: {e}")
            return None
    
    async def analyze_position(self, board: chess.Board, depth: int = 20) -> Optional[Dict[str, Any]]:
        """Analyze a position and return evaluation details."""
        if not self.is_ready or not self.process:
            logger.error("Engine not ready")
            return None
            
        try:
            limit = chess.engine.Limit(depth=depth)
            result = await self.process.analyse(board, limit)
            
            analysis = {
                "score": result.get("score", None),
                "pv": result.get("pv", []),
                "depth": result.get("depth", 0),
                "time": result.get("time", 0),
                "nodes": result.get("nodes", 0)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            return None

    async def analyze_multipv(self, board: chess.Board, time_ms: int = 100, multipv: int = 4) -> Optional[List[Dict[str, Any]]]:
        """Analyze a position and return top-N candidate moves with scores.

        Returns a list of dicts: {"move": UCI str, "score_cp": int}
        """
        if not self.is_ready or not self.process:
            logger.error("Engine not ready")
            return None
        try:
            limit = chess.engine.Limit(time=time_ms / 1000.0)
            # Request multipv variations; engine should have MultiPV option but most obey analyse param
            infos = await self.process.analyse(board, limit, multipv=max(1, int(multipv)))
            if not isinstance(infos, list):
                infos = [infos]
            out: List[Dict[str, Any]] = []
            for info in infos:
                mv = info.get("pv", [None])[0]
                score = info.get("score")
                if mv is None or score is None:
                    continue
                # Convert score to centipawns w.r.t side to move (positive = good for stm)
                try:
                    cp = score.white().score(mate_score=100000) if board.turn == chess.WHITE else score.black().score(mate_score=100000)
                except Exception:
                    # Fallback generic
                    cp = score.score(mate_score=100000) if hasattr(score, "score") else 0
                out.append({"move": mv.uci(), "score_cp": int(cp)})
            return out if out else None
        except Exception as e:
            logger.error(f"Error multipv analysis: {e}")
            return None
    
    async def is_legal_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is legal in the current position."""
        return move in board.legal_moves
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine."""
        return {
            "path": str(self.engine_path),
            "name": self.engine_path.name,
            "parameters": self.parameters,
            "time_control": self.time_control,
            "is_ready": self.is_ready
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_engine()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_engine()


class SynchronousUCIClient:
    """Synchronous wrapper for UCIClient for use in non-async contexts."""
    
    def __init__(self, engine_path: str, parameters: Dict[str, Any], time_control: str = "100ms"):
        self.async_client = UCIClient(engine_path, parameters, time_control)

    def _run(self, coro):
        """Run an async coroutine, preserving the original event loop."""
        try:
            original_loop = asyncio.get_event_loop()
        except RuntimeError:
            original_loop = None

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(original_loop)

    def start_engine(self) -> bool:
        """Start the engine synchronously."""
        try:
            return bool(self._run(self.async_client.start_engine()))
        except Exception as e:
            logger.error(f"Failed to start engine synchronously: {e}")
            return False

    def stop_engine(self):
        """Stop the engine synchronously."""
        try:
            self._run(self.async_client.stop_engine())
        except Exception as e:
            logger.error(f"Failed to stop engine synchronously: {e}")

    def get_move(self, board: chess.Board, time_ms: int = 100) -> Optional[chess.Move]:
        """Get move synchronously."""
        try:
            return self._run(self.async_client.get_move(board, time_ms))
        except Exception as e:
            logger.error(f"Failed to get move synchronously: {e}")
            return None

    def analyze_position(self, board: chess.Board, depth: int = 20) -> Optional[Dict[str, Any]]:
        """Analyze position synchronously."""
        try:
            return self._run(self.async_client.analyze_position(board, depth))
        except Exception as e:
            logger.error(f"Failed to analyze position synchronously: {e}")
            return None

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine info."""
        return self.async_client.get_engine_info()
