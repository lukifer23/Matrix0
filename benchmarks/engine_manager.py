# Enhanced Engine Manager for Matrix0 Benchmark System
"""
Advanced engine management system with automatic detection, validation, and health monitoring.
Supports Stockfish, LC0, and other UCI-compliant engines with Apple Silicon optimizations.
"""

import asyncio
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EngineInfo:
    """Comprehensive engine information."""
    name: str
    version: str
    author: str
    path: str
    protocol: str = "uci"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    estimated_elo: Optional[int] = None
    last_health_check: float = 0.0
    health_status: str = "unknown"
    supported_features: List[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    """Enhanced engine configuration with Apple Silicon optimizations."""
    name: str
    command: str
    working_dir: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    time_control: str = "30+0.3"
    estimated_rating: Optional[int] = None
    apple_silicon_optimized: bool = False

    def __post_init__(self):
        if platform.system() == "Darwin" and "arm64" in platform.machine():
            self.apple_silicon_optimized = True
            self._apply_apple_silicon_optimizations()

    def _apply_apple_silicon_optimizations(self):
        """Apply Apple Silicon-specific optimizations."""
        if self.name.lower() == "lc0":
            # LC0 Metal backend optimizations
            if "Backend" not in self.options:
                self.options["Backend"] = "metal"
            if "Blas" not in self.options:
                self.options["Blas"] = "true"
        elif self.name.lower() == "stockfish":
            # Stockfish thread optimization for Apple Silicon
            if "Threads" not in self.options:
                self.options["Threads"] = "4"


class EnhancedEngineManager:
    """Advanced engine manager with automatic detection and validation."""

    def __init__(self, config_dir: Optional[str] = None):
        self.engines: Dict[str, EngineInfo] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.config_dir = Path(config_dir) if config_dir else Path("benchmarks/engines")
        self.system = platform.system().lower()
        self.architecture = platform.machine()

        # Common engine installation paths
        self.search_paths = self._get_search_paths()

        logger.info(f"Initialized EnhancedEngineManager for {self.system} {self.architecture}")

    def _get_search_paths(self) -> List[str]:
        """Get system-specific search paths for engines."""
        paths = []

        if self.system == "darwin":  # macOS
            paths.extend([
                "/opt/homebrew/bin",      # Homebrew arm64
                "/usr/local/bin",         # Homebrew x86
                "/Applications",          # App bundles
                "/usr/bin",
                "/bin"
            ])
        elif self.system == "linux":
            paths.extend([
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
                "/opt",
                "~/.local/bin"
            ])
        elif self.system == "windows":
            paths.extend([
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                os.path.expanduser("~\\AppData\\Local"),
                os.environ.get("PATH", "").split(os.pathsep)
            ])

        # Add user PATH
        paths.extend(os.environ.get("PATH", "").split(os.pathsep))

        return list(set(paths))  # Remove duplicates

    async def discover_engines(self) -> Dict[str, EngineInfo]:
        """Automatically discover installed engines."""
        logger.info("Starting engine discovery...")

        discovered_engines = {}

        # Check for known engines
        known_engines = {
            "stockfish": ["stockfish"],
            "lc0": ["lc0", "leela"],
            "komodo": ["komodo"],
            "houdini": ["houdini"],
            "fire": ["fire"],
        }

        for engine_name, binary_names in known_engines.items():
            for binary_name in binary_names:
                engine_info = await self._discover_engine(binary_name, engine_name)
                if engine_info:
                    discovered_engines[engine_name] = engine_info
                    break

        # Check custom paths
        for search_path in self.search_paths:
            if os.path.exists(search_path):
                for item in os.listdir(search_path):
                    item_path = os.path.join(search_path, item)
                    if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                        # Try to identify as chess engine
                        engine_info = await self._identify_engine(item_path)
                        if engine_info:
                            discovered_engines[engine_info.name.lower()] = engine_info

        self.engines.update(discovered_engines)
        logger.info(f"Discovered {len(discovered_engines)} engines: {list(discovered_engines.keys())}")

        return discovered_engines

    async def _discover_engine(self, binary_name: str, engine_name: str) -> Optional[EngineInfo]:
        """Discover a specific engine by name."""
        # Check in PATH
        engine_path = shutil.which(binary_name)
        if engine_path:
            return await self._identify_engine(engine_path, engine_name)

        # Check search paths
        for search_path in self.search_paths:
            candidate_path = os.path.join(search_path, binary_name)
            if os.path.exists(candidate_path) and os.access(candidate_path, os.X_OK):
                return await self._identify_engine(candidate_path, engine_name)

        return None

    async def _identify_engine(self, engine_path: str, expected_name: Optional[str] = None) -> Optional[EngineInfo]:
        """Identify and validate a chess engine."""
        try:
            # Start engine process
            process = await asyncio.create_subprocess_exec(
                engine_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            try:
                # Send UCI initialization
                process.stdin.write("uci\n")
                await process.stdin.drain()

                # Read response with timeout
                response_lines = []
                timeout = 10.0
                start_time = time.time()

                while time.time() - start_time < timeout:
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                        line = line.strip()
                        if not line:
                            continue

                        response_lines.append(line)

                        # Check for UCI responses
                        if line == "uciok":
                            break
                        elif "id name" in line:
                            engine_name = line.replace("id name", "").strip()
                        elif "id author" in line:
                            engine_author = line.replace("id author", "").strip()

                    except asyncio.TimeoutError:
                        continue

                # Send isready to complete initialization
                process.stdin.write("isready\n")
                await process.stdin.drain()

                # Wait for readyok
                ready_timeout = 5.0
                ready_start = time.time()
                while time.time() - ready_start < ready_timeout:
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                        if line.strip() == "readyok":
                            break
                    except asyncio.TimeoutError:
                        continue

                # Clean shutdown
                process.stdin.write("quit\n")
                await process.stdin.drain()

                # Wait for process to terminate
                await asyncio.wait_for(process.wait(), timeout=2.0)

                # Parse engine information
                if response_lines:
                    engine_name = expected_name or "Unknown"
                    engine_author = "Unknown"
                    engine_version = "Unknown"

                    for line in response_lines:
                        if line.startswith("id name"):
                            engine_name = line[7:].strip()
                        elif line.startswith("id author"):
                            engine_author = line[9:].strip()

                    # Extract version from name
                    version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', engine_name)
                    if version_match:
                        engine_version = version_match.group(1)

                    # Determine capabilities
                    capabilities = {}
                    supported_features = []

                    # Check for known engine signatures
                    if "stockfish" in engine_name.lower():
                        capabilities["protocol"] = "uci"
                        capabilities["threads_support"] = True
                        capabilities["hash_support"] = True
                        supported_features.extend(["multipv", "skill_levels", "elo_limits"])
                    elif "lc0" in engine_name.lower() or "leela" in engine_name.lower():
                        capabilities["protocol"] = "uci"
                        capabilities["neural_network"] = True
                        capabilities["metal_support"] = self.architecture == "arm64"
                        supported_features.extend(["neural_network", "metal_backend"])

                    # Estimate ELO rating
                    estimated_elo = self._estimate_engine_rating(engine_name)

                    engine_info = EngineInfo(
                        name=engine_name,
                        version=engine_version,
                        author=engine_author,
                        path=engine_path,
                        protocol="uci",
                        capabilities=capabilities,
                        estimated_elo=estimated_elo,
                        last_health_check=time.time(),
                        health_status="healthy",
                        supported_features=supported_features
                    )

                    logger.info(f"Identified engine: {engine_name} v{engine_version} at {engine_path}")
                    return engine_info

            except Exception as e:
                logger.warning(f"Error identifying engine at {engine_path}: {e}")
            finally:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

        except Exception as e:
            logger.warning(f"Failed to identify engine at {engine_path}: {e}")

        return None

    def _estimate_engine_rating(self, engine_name: str) -> Optional[int]:
        """Estimate engine ELO rating based on name and version."""
        name_lower = engine_name.lower()

        # Stockfish ratings (approximate)
        if "stockfish" in name_lower:
            if "dev" in name_lower or "master" in name_lower:
                return 3800
            elif "16" in name_lower:
                return 3700
            elif "15" in name_lower:
                return 3650
            else:
                return 3500

        # LC0 ratings (approximate)
        elif "lc0" in name_lower or "leela" in name_lower:
            if "31" in name_lower:
                return 3700
            elif "30" in name_lower:
                return 3650
            else:
                return 3600

        # Komodo ratings
        elif "komodo" in name_lower:
            return 3500

        # Houdini ratings
        elif "houdini" in name_lower:
            return 3650

        return None

    async def validate_engine_health(self, engine_name: str) -> bool:
        """Validate engine health and responsiveness."""
        if engine_name not in self.engines:
            logger.error(f"Engine {engine_name} not found")
            return False

        engine_info = self.engines[engine_name]

        try:
            # Quick health check
            process = await asyncio.create_subprocess_exec(
                engine_info.path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            try:
                # Send UCI commands
                commands = ["uci", "isready", "ucinewgame", "quit"]

                for cmd in commands:
                    process.stdin.write(f"{cmd}\n")
                    await process.stdin.drain()

                    if cmd == "uci":
                        # Wait for uciok
                        timeout = 5.0
                        start = time.time()
                        while time.time() - start < timeout:
                            try:
                                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                                if line.strip() == "uciok":
                                    break
                            except asyncio.TimeoutError:
                                continue

                    elif cmd == "isready":
                        # Wait for readyok
                        timeout = 3.0
                        start = time.time()
                        while time.time() - start < timeout:
                            try:
                                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                                if line.strip() == "readyok":
                                    break
                            except asyncio.TimeoutError:
                                continue

                # Wait for process to terminate
                await asyncio.wait_for(process.wait(), timeout=2.0)

                engine_info.health_status = "healthy"
                engine_info.last_health_check = time.time()
                return True

            except Exception as e:
                logger.error(f"Health check failed for {engine_name}: {e}")
                engine_info.health_status = "unhealthy"
                return False

        except Exception as e:
            logger.error(f"Failed to start {engine_name} for health check: {e}")
            engine_info.health_status = "unavailable"
            return False

    def get_engine_config(self, engine_name: str) -> Optional[EngineConfig]:
        """Get optimized configuration for an engine."""
        if engine_name not in self.engines:
            return None

        engine_info = self.engines[engine_name]

        # Base configuration
        config = EngineConfig(
            name=engine_info.name,
            command=engine_info.path,
            estimated_rating=engine_info.estimated_elo
        )

        # Engine-specific optimizations
        if "stockfish" in engine_info.name.lower():
            config.options.update({
                "Threads": "4",
                "Hash": "512",
                "Skill Level": "20",
                "UCI_LimitStrength": "false"  # Full strength for benchmarks
            })
            config.time_control = "60+0.6"

        elif "lc0" in engine_info.name.lower():
            config.options.update({
                "Threads": "4",
                "NNCacheSize": "2000000",
                "MinibatchSize": "32",
                "MaxPrefetch": "32"
            })

            # Apple Silicon optimizations
            if self.architecture == "arm64":
                config.options.update({
                    "Backend": "metal",
                    "Blas": "true"
                })

            config.time_control = "30+0.3"

        return config

    def get_available_engines(self) -> List[str]:
        """Get list of available engines."""
        return list(self.engines.keys())

    def get_engine_info(self, engine_name: str) -> Optional[EngineInfo]:
        """Get detailed engine information."""
        return self.engines.get(engine_name)

    async def create_engine_configs(self, output_dir: Optional[str] = None) -> Dict[str, EngineConfig]:
        """Create optimized configurations for all discovered engines."""
        configs = {}

        for engine_name in self.get_available_engines():
            config = self.get_engine_config(engine_name)
            if config:
                configs[engine_name] = config

        # Save to file if requested
        if output_dir:
            output_path = Path(output_dir) / "discovered_engines.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import yaml
            with open(output_path, 'w') as f:
                # Convert to serializable format
                serializable_configs = {}
                for name, config in configs.items():
                    serializable_configs[name] = {
                        "command": config.command,
                        "working_dir": config.working_dir,
                        "options": config.options,
                        "time_control": config.time_control,
                        "estimated_rating": config.estimated_rating,
                        "apple_silicon_optimized": config.apple_silicon_optimized
                    }

                yaml.dump(serializable_configs, f, default_flow_style=False, indent=2)

            logger.info(f"Saved engine configurations to {output_path}")

        return configs

    async def validate_all_engines(self) -> Dict[str, bool]:
        """Validate health of all discovered engines."""
        results = {}
        for engine_name in self.get_available_engines():
            results[engine_name] = await self.validate_engine_health(engine_name)

        healthy_count = sum(results.values())
        logger.info(f"Engine validation complete: {healthy_count}/{len(results)} engines healthy")

        return results


# Global instance
enhanced_engine_manager = EnhancedEngineManager()


async def discover_and_validate_engines():
    """Convenience function to discover and validate all engines."""
    await enhanced_engine_manager.discover_engines()
    await enhanced_engine_manager.validate_all_engines()
    return enhanced_engine_manager


if __name__ == "__main__":
    # Example usage
    async def main():
        print("Discovering chess engines...")
        await discover_and_validate_engines()

        print(f"\nAvailable engines: {enhanced_engine_manager.get_available_engines()}")

        for engine_name in enhanced_engine_manager.get_available_engines():
            info = enhanced_engine_manager.get_engine_info(engine_name)
            config = enhanced_engine_manager.get_engine_config(engine_name)

            print(f"\n{engine_name}:")
            print(f"  Path: {info.path}")
            print(f"  Version: {info.version}")
            print(f"  Author: {info.author}")
            print(f"  Estimated ELO: {info.estimated_elo}")
            print(f"  Health: {info.health_status}")
            print(f"  Features: {', '.join(info.supported_features)}")
            print(f"  Apple Silicon Optimized: {config.apple_silicon_optimized if config else False}")

    asyncio.run(main())
