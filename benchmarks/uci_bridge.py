# UCI Engine Bridge
"""
UCI (Universal Chess Interface) communication bridge for chess engines.
Handles communication with Stockfish, lc0, and other UCI-compliant engines.
"""

import subprocess
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class UCIEngine:
    """UCI-compliant chess engine interface."""

    def __init__(self, config):
        self.name = config.name
        self.command = config.command
        self.working_dir = config.working_dir
        self.options = config.options or {}

        self.process = None
        self.stdin = None
        self.stdout = None
        self.stderr = None

        self.is_ready = False
        self.engine_info = {}

    def start(self) -> bool:
        """Start the UCI engine process."""
        try:
            if self.working_dir:
                working_dir = Path(self.working_dir)
            else:
                working_dir = None

            logger.info(f"Starting UCI engine: {self.name} ({self.command})")

            self.process = subprocess.Popen(
                self.command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                bufsize=1,
                universal_newlines=True
            )

            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            self.stderr = self.process.stderr

            # Initialize UCI protocol
            if not self._initialize_uci():
                logger.error(f"Failed to initialize UCI protocol for {self.name}")
                self.stop()
                return False

            # Set options
            self._set_options()

            # Wait for ready
            if not self._wait_ready():
                logger.error(f"Engine {self.name} failed to become ready")
                return False

            self.is_ready = True
            logger.info(f"UCI engine {self.name} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start UCI engine {self.name}: {e}")
            return False

    def stop(self):
        """Stop the UCI engine process."""
        if self.process:
            logger.info(f"Stopping UCI engine: {self.name}")
            self._send_command("quit")
            time.sleep(0.1)  # Give time for clean shutdown

            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

            self.process = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self.is_ready = False

    def _send_command(self, command: str):
        """Send a command to the engine."""
        if self.stdin:
            logger.debug(f"UCI -> {self.name}: {command}")
            try:
                self.stdin.write(command + "\n")
                self.stdin.flush()
            except Exception as e:
                logger.error(f"Failed to send command '{command}' to {self.name}: {e}")

    def _read_response(self, timeout: float = 1.0) -> List[str]:
        """Read response from the engine."""
        if not self.stdout:
            return []

        lines = []
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                line = self.stdout.readline().strip()
                if line:
                    logger.debug(f"UCI <- {self.name}: {line}")
                    lines.append(line)

                    # Check for specific responses
                    if line == "readyok":
                        break
                    if "bestmove" in line:
                        break
                    if line.startswith("info") and "score" in line:
                        continue  # Continue reading info lines
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
        except Exception as e:
            logger.error(f"Error reading from {self.name}: {e}")

        return lines

    def _initialize_uci(self) -> bool:
        """Initialize UCI protocol."""
        self._send_command("uci")

        lines = self._read_response(timeout=2.0)
        uciok = False

        for line in lines:
            if line.startswith("id name"):
                self.engine_info["name"] = line[7:].strip()
            elif line.startswith("id author"):
                self.engine_info["author"] = line[9:].strip()
            elif line.startswith("uciok"):
                uciok = True

        if not uciok:
            logger.error(f"No 'uciok' received from {self.name}")
            return False

        return True

    def _set_options(self):
        """Set UCI options."""
        for option_name, option_value in self.options.items():
            command = f"setoption name {option_name} value {option_value}"
            self._send_command(command)
            time.sleep(0.05)  # Small delay between option sets

    def _wait_ready(self) -> bool:
        """Wait for engine to be ready."""
        self._send_command("isready")

        lines = self._read_response(timeout=5.0)

        for line in lines:
            if line == "readyok":
                return True

        return False

    def new_game(self):
        """Start a new game."""
        self._send_command("ucinewgame")

    def set_position(self, fen: str, moves: Optional[List[str]] = None):
        """Set the current position."""
        if moves:
            moves_str = " ".join(moves)
            command = f"position fen {fen} moves {moves_str}"
        else:
            command = f"position fen {fen}"

        self._send_command(command)

    def go(self, time_control: str = "infinite") -> Tuple[str, float]:
        """Start engine analysis and get best move."""
        if time_control == "infinite":
            command = "go infinite"
        else:
            # Parse time control (e.g., "60+0.6" -> 60 seconds + 0.6 increment)
            match = re.match(r"(\d+)\+([\d.]+)", time_control)
            if match:
                base_time = int(match.group(1)) * 1000  # Convert to milliseconds
                increment = int(float(match.group(2)) * 1000)
                command = f"go wtime {base_time} btime {base_time} winc {increment} binc {increment}"
            else:
                command = "go infinite"

        start_time = time.time()
        self._send_command(command)

        lines = self._read_response(timeout=30.0)

        end_time = time.time()
        elapsed = end_time - start_time

        # Extract best move
        best_move = None
        for line in reversed(lines):  # Check from the end
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        if best_move is None:
            logger.warning(f"No bestmove found for {self.name}")
            best_move = "0000"  # Null move as fallback

        return best_move, elapsed

    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return self.engine_info.copy()

    def is_alive(self) -> bool:
        """Check if the engine process is still alive."""
        if self.process:
            return self.process.poll() is None
        return False


class EngineManager:
    """Manages multiple UCI engines."""

    def __init__(self):
        self.engines: Dict[str, UCIEngine] = {}

    def add_engine(self, config) -> bool:
        """Add and start a UCI engine."""
        engine = UCIEngine(config)

        if engine.start():
            self.engines[config.name] = engine
            return True
        else:
            logger.error(f"Failed to add engine: {config.name}")
            return False

    def remove_engine(self, name: str):
        """Remove and stop a UCI engine."""
        if name in self.engines:
            self.engines[name].stop()
            del self.engines[name]

    def get_engine(self, name: str) -> Optional[UCIEngine]:
        """Get an engine by name."""
        return self.engines.get(name)

    def list_engines(self) -> List[str]:
        """List available engines."""
        return list(self.engines.keys())

    def stop_all(self):
        """Stop all engines."""
        for engine in self.engines.values():
            engine.stop()
        self.engines.clear()


# Global engine manager instance
engine_manager = EngineManager()
