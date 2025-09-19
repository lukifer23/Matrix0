# Tournament System for Matrix0 Benchmarks
"""
Advanced tournament system for competitive evaluation of chess engines.
Supports round-robin, single-elimination, and Swiss tournament formats.
"""

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TournamentFormat(Enum):
    """Supported tournament formats."""
    ROUND_ROBIN = "round_robin"
    SINGLE_ELIMINATION = "single_elimination"
    SWISS = "swiss"
    DOUBLE_ROUND_ROBIN = "double_round_robin"


class TournamentResult(Enum):
    """Possible game results."""
    WHITE_WIN = "1-0"
    BLACK_WIN = "0-1"
    DRAW = "1/2-1/2"


@dataclass
class GameResult:
    """Result of a single game."""
    white_engine: str
    black_engine: str
    result: TournamentResult
    game_time: float
    move_count: int
    pgn_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedTimeControl:
    """Container for parsed time control settings."""

    original: str
    time_seconds: Optional[float]
    increment_seconds: float = 0.0

    def limit_kwargs(self) -> Dict[str, float]:
        """Return keyword arguments compatible with ``chess.engine.Limit``."""
        kwargs: Dict[str, float] = {}

        if self.time_seconds is not None:
            kwargs["time"] = self.time_seconds

        if self.increment_seconds:
            kwargs["white_inc"] = self.increment_seconds
            kwargs["black_inc"] = self.increment_seconds

        return kwargs

    def to_limit(self) -> "chess.engine.Limit":
        """Instantiate a ``chess.engine.Limit`` for the parsed settings."""
        import chess.engine

        return chess.engine.Limit(**self.limit_kwargs())


def _parse_time_fragment(value: Union[str, float, int]) -> float:
    """Parse a single duration fragment into seconds."""

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Unsupported time control fragment type: {type(value)!r}")

    text = value.strip().lower()

    if not text:
        raise ValueError("Empty time control fragment")

    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0

    if text.endswith("s"):
        return float(text[:-1])

    return float(text)


def parse_time_control(time_control: Union[str, float, int, ParsedTimeControl]) -> ParsedTimeControl:
    """Parse a time control specification into seconds.

    Supported formats include:

    - ``"30+0.3"`` – base time plus per-move increment (in seconds)
    - ``"100ms"`` – milliseconds
    - ``"60s"`` – seconds with explicit unit
    - raw numeric seconds (``30`` or ``30.0``)
    """

    if isinstance(time_control, ParsedTimeControl):
        return time_control

    if isinstance(time_control, (int, float)):
        return ParsedTimeControl(
            original=str(time_control),
            time_seconds=float(time_control),
            increment_seconds=0.0,
        )

    if not isinstance(time_control, str):
        raise ValueError(f"Unsupported time control type: {type(time_control)!r}")

    original = time_control
    cleaned = time_control.strip()

    if "+" in cleaned:
        base_text, increment_text = cleaned.split("+", 1)
        base_seconds = _parse_time_fragment(base_text)
        increment_seconds = _parse_time_fragment(increment_text)
        return ParsedTimeControl(
            original=original,
            time_seconds=base_seconds,
            increment_seconds=increment_seconds,
        )

    time_seconds = _parse_time_fragment(cleaned)
    return ParsedTimeControl(
        original=original,
        time_seconds=time_seconds,
        increment_seconds=0.0,
    )


@dataclass
class EngineStanding:
    """Tournament standing for an engine."""
    engine_name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0  # 1.0 for win, 0.5 for draw, 0.0 for loss
    buchholz_score: float = 0.0  # Sum of opponents' scores
    sonneborn_berger_score: float = 0.0  # Weighted score based on opponents
    elo_performance: Optional[float] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

    @property
    def score_percentage(self) -> float:
        """Calculate score as percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.points / self.games_played) * 100.0


@dataclass
class TournamentConfig:
    """Configuration for tournament execution."""
    name: str
    format: TournamentFormat
    engines: List[str]
    num_games_per_pairing: int = 1
    time_control: Union[str, float, int] = "30+0.3"
    max_moves: int = 200
    concurrency: int = 2
    random_openings: bool = True
    opening_plies: int = 8
    output_dir: str = "benchmarks/results"
    save_pgns: bool = True
    calculate_ratings: bool = True


@dataclass
class TournamentRound:
    """A single round of tournament games."""
    round_number: int
    pairings: List[Tuple[str, str]]  # (white, black) engine pairs
    results: List[GameResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Check if all games in the round are complete."""
        return len(self.results) == len(self.pairings)


class Tournament:
    """Advanced tournament system for chess engine evaluation."""

    def __init__(self, config: TournamentConfig):
        self.config = config
        self.standings: Dict[str, EngineStanding] = {}
        self.rounds: List[TournamentRound] = []
        self.completed_games: List[GameResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._parsed_time_control = parse_time_control(config.time_control)

        # Initialize standings
        for engine in config.engines:
            self.standings[engine] = EngineStanding(engine_name=engine)

        logger.info(f"Tournament '{config.name}' initialized with {len(config.engines)} engines")

    async def run_tournament(self) -> Dict[str, Any]:
        """Run the complete tournament."""
        self.start_time = time.time()
        logger.info(f"Starting tournament: {self.config.name}")

        try:
            if self.config.format == TournamentFormat.ROUND_ROBIN:
                await self._run_round_robin()
            elif self.config.format == TournamentFormat.DOUBLE_ROUND_ROBIN:
                await self._run_double_round_robin()
            elif self.config.format == TournamentFormat.SINGLE_ELIMINATION:
                await self._run_single_elimination()
            elif self.config.format == TournamentFormat.SWISS:
                await self._run_swiss()
            else:
                raise ValueError(f"Unsupported tournament format: {self.config.format}")

            self.end_time = time.time()
            self._calculate_final_standings()

            logger.info(f"Tournament completed in {self.duration:.1f} seconds")
            return self.get_tournament_results()

        except Exception as e:
            logger.error(f"Tournament failed: {e}")
            raise

    async def _run_round_robin(self):
        """Run a round-robin tournament."""
        logger.info("Running round-robin tournament")

        # Generate all possible pairings
        pairings = []
        engines = self.config.engines.copy()

        for i, engine1 in enumerate(engines):
            for engine2 in engines[i + 1:]:
                # Each pairing plays both colors
                pairings.extend([
                    (engine1, engine2),
                    (engine2, engine1)
                ])

        # Run all games
        await self._run_pairings(pairings)

    async def _run_double_round_robin(self):
        """Run a double round-robin tournament."""
        logger.info("Running double round-robin tournament")

        # Generate all possible pairings (twice)
        pairings = []
        engines = self.config.engines.copy()

        for round_num in range(2):
            for i, engine1 in enumerate(engines):
                for engine2 in engines[i + 1:]:
                    if round_num == 0:
                        pairings.extend([(engine1, engine2), (engine2, engine1)])
                    else:
                        pairings.extend([(engine2, engine1), (engine1, engine2)])

        await self._run_pairings(pairings)

    async def _run_single_elimination(self):
        """Run a single-elimination tournament."""
        logger.info("Running single-elimination tournament")

        # Start with all engines
        remaining_engines = self.config.engines.copy()
        random.shuffle(remaining_engines)

        round_num = 1

        while len(remaining_engines) > 1:
            logger.info(f"Round {round_num}: {len(remaining_engines)} engines remaining")

            # Pair engines for this round
            pairings = []
            winners = []

            # Handle odd number of engines
            if len(remaining_engines) % 2 == 1:
                # One engine gets a bye
                bye_engine = remaining_engines.pop()
                winners.append(bye_engine)
                logger.info(f"Engine {bye_engine} gets bye to next round")

            # Create pairings
            for i in range(0, len(remaining_engines), 2):
                pairings.append((remaining_engines[i], remaining_engines[i + 1]))

            # Play games for this round
            round_results = await self._run_pairings(pairings, return_results=True)

            # Determine winners
            for white, black, result in round_results:
                if result == TournamentResult.WHITE_WIN:
                    winners.append(white)
                elif result == TournamentResult.BLACK_WIN:
                    winners.append(black)
                else:
                    # Draw - play again or random winner
                    winner = random.choice([white, black])
                    winners.append(winner)
                    logger.info(f"Draw between {white} and {black}, {winner} advances randomly")

            remaining_engines = winners
            round_num += 1

        # Final winner
        if remaining_engines:
            logger.info(f"Tournament winner: {remaining_engines[0]}")

    async def _run_swiss(self):
        """Run a Swiss-system tournament."""
        logger.info("Running Swiss tournament")

        # Swiss tournament with multiple rounds
        num_rounds = math.ceil(math.log2(len(self.config.engines)))

        for round_num in range(1, num_rounds + 1):
            logger.info(f"Swiss round {round_num}")

            # Pair engines based on current standings
            pairings = self._generate_swiss_pairings(round_num)

            if not pairings:
                logger.info("No more pairings possible in Swiss tournament")
                break

            await self._run_pairings(pairings)

    def _generate_swiss_pairings(self, round_num: int) -> List[Tuple[str, str]]:
        """Generate pairings for Swiss tournament round."""
        # Sort engines by current score
        sorted_engines = sorted(
            self.standings.keys(),
            key=lambda x: (
                self.standings[x].points,
                self.standings[x].buchholz_score,
                self.standings[x].sonneborn_berger_score
            ),
            reverse=True
        )

        pairings = []
        used_engines = set()

        # Simple Swiss pairing algorithm
        for i, engine1 in enumerate(sorted_engines):
            if engine1 in used_engines:
                continue

            # Find opponent with similar score
            for j, engine2 in enumerate(sorted_engines[i + 1:], i + 1):
                if engine2 in used_engines:
                    continue

                # Check if these engines have played before (avoid rematches)
                if not self._engines_played_before(engine1, engine2):
                    pairings.append((engine1, engine2))
                    used_engines.update([engine1, engine2])
                    break

        return pairings

    def _engines_played_before(self, engine1: str, engine2: str) -> bool:
        """Check if two engines have played each other."""
        for result in self.completed_games:
            if ((result.white_engine == engine1 and result.black_engine == engine2) or
                (result.white_engine == engine2 and result.black_engine == engine1)):
                return True
        return False

    async def _run_pairings(self, pairings: List[Tuple[str, str]],
                          return_results: bool = False) -> List[Tuple[str, str, TournamentResult]]:
        """Run games for a set of pairings."""
        logger.info(f"Running {len(pairings)} pairings")

        results = []

        # Create concurrent tasks for games
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def run_game(white: str, black: str) -> GameResult:
            async with semaphore:
                return await self._play_game(white, black)

        # Run all games concurrently
        tasks = [run_game(white, black) for white, black in pairings]
        game_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(game_results):
            white, black = pairings[i]

            if isinstance(result, Exception):
                logger.error(f"Game between {white} and {black} failed: {result}")
                # Record as draw on error
                game_result = GameResult(
                    white_engine=white,
                    black_engine=black,
                    result=TournamentResult.DRAW,
                    game_time=0.0,
                    move_count=0
                )
            else:
                game_result = result

            self.completed_games.append(game_result)
            self._update_standings(game_result)

            if return_results:
                results.append((white, black, game_result.result))

        return results

    def _create_engine_limit(self) -> "chess.engine.Limit":
        """Create a chess engine limit using the parsed time control."""

        return self._parsed_time_control.to_limit()

    async def _play_game(self, white_engine: str, black_engine: str) -> GameResult:
        """Play a single game between two engines."""
        logger.info(f"Playing game: {white_engine} (White) vs {black_engine} (Black)")

        start_time = time.time()

        # Import here to avoid circular imports
        from azchess.config import Config, select_device
        from azchess.model import PolicyValueNet
        from azchess.mcts import MCTS, MCTSConfig
        from azchess.encoding import encode_board

        try:
            # Initialize engines if needed
            matrix0_model = None
            matrix0_mcts = None
            stockfish_engine = None

            # Load Matrix0 if one of the engines is matrix0
            if white_engine == "matrix0" or black_engine == "matrix0":
                cfg = Config.load("config.yaml")
                device = select_device(cfg.training().get("device", "cpu"))
                model = PolicyValueNet.from_config(cfg.model())
                model.eval()

                eval_cfg = cfg.eval()
                mcts_cfg = MCTSConfig(
                    num_simulations=eval_cfg.get("num_simulations", 200),
                    cpuct=eval_cfg.get("cpuct", 1.5)
                )
                mcts = MCTS(model, mcts_cfg, device)

                matrix0_model = model
                matrix0_mcts = mcts

            # Load Stockfish if one of the engines is stockfish
            if white_engine == "stockfish" or black_engine == "stockfish":
                try:
                    import chess.engine
                    stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                except Exception as e:
                    logger.error(f"Failed to load Stockfish: {e}")
                    # Return draw if Stockfish unavailable
                    return GameResult(
                        white_engine=white_engine,
                        black_engine=black_engine,
                        result=TournamentResult.DRAW,
                        game_time=time.time() - start_time,
                        move_count=0,
                        metadata={"error": "Stockfish unavailable"}
                    )

            # Play the game
            board = chess.Board()
            moves = []
            move_count = 0

            while not board.is_game_over() and move_count < self.config.max_moves:
                current_engine = white_engine if board.turn == chess.WHITE else black_engine

                if current_engine == "matrix0":
                    # Matrix0 move
                    visits, pi, v = matrix0_mcts.run(board)
                    move = max(visits.items(), key=lambda kv: kv[1])[0]
                elif current_engine == "stockfish" and stockfish_engine:
                    # Stockfish move
                    limit = self._create_engine_limit()
                    result = stockfish_engine.play(board, limit)
                    move = result.move
                else:
                    # Unknown engine - make random legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                    else:
                        break

                if move:
                    board.push(move)
                    moves.append(move.uci())
                    move_count += 1

            # Determine result
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    result = TournamentResult.BLACK_WIN
                else:
                    result = TournamentResult.WHITE_WIN
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                result = TournamentResult.DRAW
            else:
                # Game hit max moves or other termination
                result = TournamentResult.DRAW

            game_time = time.time() - start_time

            # Clean up engines
            if stockfish_engine:
                try:
                    stockfish_engine.quit()
                except:
                    pass

            game_result = GameResult(
                white_engine=white_engine,
                black_engine=black_engine,
                result=result,
                game_time=game_time,
                move_count=move_count,
                metadata={
                    "tournament_format": self.config.format.value,
                    "time_control": {
                        "original": self._parsed_time_control.original,
                        "time_seconds": self._parsed_time_control.time_seconds,
                        "increment_seconds": self._parsed_time_control.increment_seconds,
                    },
                    "moves": moves
                }
            )

            logger.info(f"Game completed: {white_engine} vs {black_engine} -> {result.value} ({move_count} moves, {game_time:.1f}s)")
            return game_result

        except Exception as e:
            logger.error(f"Error playing game {white_engine} vs {black_engine}: {e}")
            # Return draw on error
            return GameResult(
                white_engine=white_engine,
                black_engine=black_engine,
                result=TournamentResult.DRAW,
                game_time=time.time() - start_time,
                move_count=0,
                metadata={"error": str(e)}
            )

    def _update_standings(self, game_result: GameResult):
        """Update tournament standings after a game."""
        white_standing = self.standings[game_result.white_engine]
        black_standing = self.standings[game_result.black_engine]

        white_standing.games_played += 1
        black_standing.games_played += 1

        if game_result.result == TournamentResult.WHITE_WIN:
            white_standing.wins += 1
            white_standing.points += 1.0
            black_standing.losses += 1
        elif game_result.result == TournamentResult.BLACK_WIN:
            black_standing.wins += 1
            black_standing.points += 1.0
            white_standing.losses += 1
        else:  # Draw
            white_standing.draws += 1
            black_standing.draws += 1
            white_standing.points += 0.5
            black_standing.points += 0.5

    def _calculate_final_standings(self):
        """Calculate final tournament standings and ratings."""
        logger.info("Calculating final tournament standings")

        # Calculate Buchholz scores (sum of opponents' points)
        for engine_name, standing in self.standings.items():
            opponent_points = 0.0
            opponent_count = 0

            for game in self.completed_games:
                if game.white_engine == engine_name:
                    opponent_points += self.standings[game.black_engine].points
                    opponent_count += 1
                elif game.black_engine == engine_name:
                    opponent_points += self.standings[game.white_engine].points
                    opponent_count += 1

            if opponent_count > 0:
                standing.buchholz_score = opponent_points

        # Calculate Sonneborn-Berger scores
        for engine_name, standing in self.standings.items():
            sb_score = 0.0

            for game in self.completed_games:
                if game.white_engine == engine_name:
                    opponent_standing = self.standings[game.black_engine]
                    if game.result == TournamentResult.WHITE_WIN:
                        sb_score += opponent_standing.points
                    elif game.result == TournamentResult.DRAW:
                        sb_score += opponent_standing.points * 0.5
                elif game.black_engine == engine_name:
                    opponent_standing = self.standings[game.white_engine]
                    if game.result == TournamentResult.BLACK_WIN:
                        sb_score += opponent_standing.points
                    elif game.result == TournamentResult.DRAW:
                        sb_score += opponent_standing.points * 0.5

            standing.sonneborn_berger_score = sb_score

    def get_tournament_results(self) -> Dict[str, Any]:
        """Get comprehensive tournament results."""
        # Sort standings by points, then Buchholz, then Sonneborn-Berger
        sorted_standings = sorted(
            self.standings.values(),
            key=lambda x: (x.points, x.buchholz_score, x.sonneborn_berger_score),
            reverse=True
        )

        results = {
            "tournament_name": self.config.name,
            "format": self.config.format.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_games": len(self.completed_games),
            "engines": [s.engine_name for s in sorted_standings],
            "standings": [],

            "final_rankings": [],
            "game_results": [],
            "statistics": self._calculate_tournament_statistics()
        }

        # Add detailed standings
        for standing in sorted_standings:
            standing_data = {
                "engine": standing.engine_name,
                "rank": sorted_standings.index(standing) + 1,
                "games_played": standing.games_played,
                "wins": standing.wins,
                "losses": standing.losses,
                "draws": standing.draws,
                "points": standing.points,
                "win_rate": standing.win_rate,
                "score_percentage": standing.score_percentage,
                "buchholz_score": standing.buchholz_score,
                "sonneborn_berger_score": standing.sonneborn_berger_score
            }
            results["standings"].append(standing_data)
            results["final_rankings"].append(standing.engine_name)

        # Add game results
        for game in self.completed_games:
            game_data = {
                "white": game.white_engine,
                "black": game.black_engine,
                "result": game.result.value,
                "game_time": game.game_time,
                "move_count": game.move_count,
                "metadata": game.metadata
            }
            results["game_results"].append(game_data)

        return results

    def _calculate_tournament_statistics(self) -> Dict[str, Any]:
        """Calculate tournament statistics."""
        if not self.completed_games:
            return {}

        game_times = [g.game_time for g in self.completed_games]
        move_counts = [g.move_count for g in self.completed_games]

        stats = {
            "avg_game_time": np.mean(game_times),
            "min_game_time": np.min(game_times),
            "max_game_time": np.max(game_time),
            "std_game_time": np.std(game_times),

            "avg_moves_per_game": np.mean(move_counts),
            "min_moves_per_game": np.min(move_counts),
            "max_moves_per_game": np.max(move_counts),
            "std_moves_per_game": np.std(move_counts),

            "total_white_wins": sum(1 for g in self.completed_games if g.result == TournamentResult.WHITE_WIN),
            "total_black_wins": sum(1 for g in self.completed_games if g.result == TournamentResult.BLACK_WIN),
            "total_draws": sum(1 for g in self.completed_games if g.result == TournamentResult.DRAW),

            "first_move_advantage": self._calculate_first_move_advantage()
        }

        return stats

    def _calculate_first_move_advantage(self) -> float:
        """Calculate the advantage of playing first."""
        white_wins = sum(1 for g in self.completed_games if g.result == TournamentResult.WHITE_WIN)
        black_wins = sum(1 for g in self.completed_games if g.result == TournamentResult.BLACK_WIN)
        draws = sum(1 for g in self.completed_games if g.result == TournamentResult.DRAW)

        total_games = len(self.completed_games)
        if total_games == 0:
            return 0.0

        # Calculate win rate for white pieces
        white_score = white_wins + 0.5 * draws
        return (white_score / total_games) - 0.5  # Subtract 0.5 for no advantage

    @property
    def duration(self) -> float:
        """Get tournament duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


async def run_tournament(config: TournamentConfig) -> Dict[str, Any]:
    """Convenience function to run a tournament."""
    tournament = Tournament(config)
    return await tournament.run_tournament()


def create_tournament_config(
    name: str,
    engines: List[str],
    format: str = "round_robin",
    num_games_per_pairing: int = 1,
    time_control: Union[str, float, int] = "30+0.3"
) -> TournamentConfig:
    """Create a tournament configuration."""
    return TournamentConfig(
        name=name,
        format=TournamentFormat(format),
        engines=engines,
        num_games_per_pairing=num_games_per_pairing,
        time_control=time_control
    )
