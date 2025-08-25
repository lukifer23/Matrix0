"""Engine manager for coordinating external chess engines."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .uci_bridge import SynchronousUCIClient, UCIClient

logger = logging.getLogger(__name__)


class EngineManager:
    """Manages external chess engines and provides unified interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines: Dict[str, UCIClient] = {}
        self.ratings: Dict[str, float] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.engine_configs = config.get("engines", {})
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize engine configurations."""
        for engine_name, engine_config in self.engine_configs.items():
            if not engine_config.get("enabled", False):
                continue
                
            if engine_config.get("type") == "internal":
                # Internal Matrix0 engine - no UCI client needed
                self.health_status[engine_name] = {
                    "status": "internal",
                    "last_check": time.time(),
                    "is_healthy": True
                }
                continue
            
            # External engine
            engine_path = engine_config.get("path")
            parameters = engine_config.get("parameters", {})
            time_control = engine_config.get("time_control", "100ms")
            
            if not engine_path:
                logger.warning(f"Engine {engine_name} has no path specified")
                continue
            
            # Initialize rating estimate
            self.ratings[engine_name] = engine_config.get("estimated_rating", 1500.0)
            
            # Initialize health status
            self.health_status[engine_name] = {
                "status": "stopped",
                "last_check": time.time(),
                "is_healthy": False,
                "startup_attempts": 0,
                "last_error": None
            }
    
    async def start_engine(self, engine_name: str) -> bool:
        """Start a specific engine."""
        if engine_name not in self.engine_configs:
            logger.error(f"Unknown engine: {engine_name}")
            return False
        
        engine_config = self.engine_configs[engine_name]
        
        if engine_config.get("type") == "internal":
            logger.info(f"Engine {engine_name} is internal, no startup needed")
            return True
        
        if engine_name in self.engines and self.engines[engine_name].is_ready:
            logger.info(f"Engine {engine_name} is already running")
            return True
        
        try:
            engine_path = engine_config.get("path")
            parameters = engine_config.get("parameters", {})
            time_control = engine_config.get("time_control", "100ms")
            
            client = UCIClient(engine_path, parameters, time_control)
            success = await client.start_engine()
            
            if success:
                self.engines[engine_name] = client
                self.health_status[engine_name].update({
                    "status": "running",
                    "is_healthy": True,
                    "last_check": time.time(),
                    "last_error": None
                })
                logger.info(f"Engine {engine_name} started successfully")
                return True
            else:
                self.health_status[engine_name].update({
                    "status": "failed",
                    "is_healthy": False,
                    "startup_attempts": self.health_status[engine_name].get("startup_attempts", 0) + 1,
                    "last_error": "Startup failed"
                })
                return False
                
        except Exception as e:
            self.health_status[engine_name].update({
                "status": "error",
                "is_healthy": False,
                "startup_attempts": self.health_status[engine_name].get("startup_attempts", 0) + 1,
                "last_error": str(e)
            })
            logger.error(f"Error starting engine {engine_name}: {e}")
            return False
    
    async def start_all_engines(self) -> Dict[str, bool]:
        """Start all enabled external engines."""
        results = {}
        for engine_name in self.engine_configs:
            if self.engine_configs[engine_name].get("enabled", False):
                results[engine_name] = await self.start_engine(engine_name)
        return results
    
    async def stop_engine(self, engine_name: str):
        """Stop a specific engine."""
        if engine_name in self.engines:
            await self.engines[engine_name].stop_engine()
            del self.engines[engine_name]
            self.health_status[engine_name].update({
                "status": "stopped",
                "is_healthy": False,
                "last_check": time.time()
            })
            logger.info(f"Engine {engine_name} stopped")
    
    async def stop_all_engines(self):
        """Stop all running engines."""
        for engine_name in list(self.engines.keys()):
            await self.stop_engine(engine_name)
    
    async def restart_engine(self, engine_name: str) -> bool:
        """Restart a specific engine."""
        await self.stop_engine(engine_name)
        return await self.start_engine(engine_name)
    
    def get_engine(self, engine_name: str) -> Optional[UCIClient]:
        """Get a running engine client."""
        return self.engines.get(engine_name)
    
    def is_engine_healthy(self, engine_name: str) -> bool:
        """Check if an engine is healthy."""
        if engine_name not in self.health_status:
            return False
        return self.health_status[engine_name].get("is_healthy", False)
    
    async def check_engine_health(self, engine_name: str) -> bool:
        """Perform health check on an engine."""
        if engine_name not in self.engine_configs:
            return False
        
        engine_config = self.engine_configs[engine_name]
        
        if engine_config.get("type") == "internal":
            # Internal engine is always healthy
            self.health_status[engine_name].update({
                "status": "internal",
                "last_check": time.time(),
                "is_healthy": True
            })
            return True
        
        if engine_name not in self.engines:
            return False
        
        try:
            # Simple health check - try to analyze starting position
            import chess
            board = chess.Board()
            
            # Quick analysis to test engine responsiveness
            analysis = await self.engines[engine_name].analyze_position(board, depth=1)
            
            if analysis:
                self.health_status[engine_name].update({
                    "status": "running",
                    "is_healthy": True,
                    "last_check": time.time(),
                    "last_error": None
                })
                return True
            else:
                self.health_status[engine_name].update({
                    "status": "unresponsive",
                    "is_healthy": False,
                    "last_check": time.time(),
                    "last_error": "Engine unresponsive"
                })
                return False
                
        except Exception as e:
            self.health_status[engine_name].update({
                "status": "error",
                "is_healthy": False,
                "last_check": time.time(),
                "last_error": str(e)
            })
            logger.error(f"Health check failed for engine {engine_name}: {e}")
            return False
    
    async def check_all_engines_health(self) -> Dict[str, bool]:
        """Check health of all engines."""
        results = {}
        for engine_name in self.engine_configs:
            if self.engine_configs[engine_name].get("enabled", False):
                results[engine_name] = await self.check_engine_health(engine_name)
        return results
    
    def estimate_strength(self, engine_name: str) -> float:
        """Get estimated strength rating for an engine."""
        return self.ratings.get(engine_name, 1500.0)
    
    def select_training_partner(self, target_strength: float, exclude_engines: List[str] = None) -> Optional[str]:
        """Select an appropriate training partner based on target strength."""
        if exclude_engines is None:
            exclude_engines = []
        
        available_engines = []
        for engine_name, engine_config in self.engine_configs.items():
            if (engine_config.get("enabled", False) and 
                engine_name not in exclude_engines and
                self.is_engine_healthy(engine_name)):
                
                rating = self.estimate_strength(engine_name)
                # Calculate distance from target strength
                distance = abs(rating - target_strength)
                available_engines.append((engine_name, distance, rating))
        
        if not available_engines:
            return None
        
        # Sort by distance to target strength, then by rating
        available_engines.sort(key=lambda x: (x[1], x[2]))
        
        # Return the closest match
        return available_engines[0][0]
    
    def get_engine_info(self, engine_name: str) -> Dict[str, Any]:
        """Get comprehensive information about an engine."""
        if engine_name not in self.engine_configs:
            return {}
        
        engine_config = self.engine_configs[engine_name]
        health = self.health_status.get(engine_name, {})
        
        info = {
            "name": engine_name,
            "type": engine_config.get("type", "external"),
            "enabled": engine_config.get("enabled", False),
            "path": engine_config.get("path", ""),
            "parameters": engine_config.get("parameters", {}),
            "time_control": engine_config.get("time_control", "100ms"),
            "estimated_rating": self.ratings.get(engine_name, 1500.0),
            "health": health.copy()
        }
        
        if engine_name in self.engines:
            info["client_info"] = self.engines[engine_name].get_engine_info()
        
        return info
    
    def get_all_engines_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all engines."""
        return {name: self.get_engine_info(name) for name in self.engine_configs}
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_all_engines()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_all_engines()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
