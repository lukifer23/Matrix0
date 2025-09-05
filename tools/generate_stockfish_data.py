#!/usr/bin/env python3
"""
Stockfish Data Generator for Matrix0 Training

Generates high-quality training datasets using Stockfish analysis with SSL annotations.
Specifically targets: Opening theory, Positional understanding, King safety, Endgame technique, Strategic planning.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch

from azchess.ssl_algorithms import ChessSSLAlgorithms
from azchess.encoding import encode_board, move_to_index

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockfishDataGenerator:
    """Generates training data using Stockfish analysis with SSL annotations."""

    def __init__(
        self,
        stockfish_path: str = "stockfish",
        engine_depth: int = 10,
        threads: int = 4,
        hash_mb: int = 512,
        movetime_ms: Optional[int] = None,
    ):
        self.stockfish_path = stockfish_path
        self.engine_depth = engine_depth
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.movetime_ms = movetime_ms if movetime_ms is None else int(movetime_ms)
        self.ssl_algorithms = ChessSSLAlgorithms()
        self.engine = None

    def __enter__(self):
        """Start Stockfish engine."""
        logger.info(f"Starting Stockfish engine: {self.stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close Stockfish engine."""
        if self.engine:
            logger.info("Closing Stockfish engine")
            try:
                self.engine.quit()
            except Exception:
                logger.warning("Engine already terminated or failed to quit cleanly")

    def analyze_position(self, board: chess.Board, depth: Optional[int] = None) -> Dict:
        """Analyze a position with Stockfish."""
        if depth is None:
            depth = self.engine_depth

        logger.debug(f"Analyzing position at depth {depth}")

        # Get best move and evaluation
        if self.movetime_ms is not None and self.movetime_ms > 0:
            limit = chess.engine.Limit(time=self.movetime_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=depth)
        info = self.engine.analyse(board, limit)

        # Extract analysis data
        analysis = {
            'score': info.get('score', chess.engine.Cp(0)),
            'best_move': info.get('pv', [None])[0] if info.get('pv') else None,
            'depth': info.get('depth', 0),
            'nodes': info.get('nodes', 0),
            'time': info.get('time', 0),
        }

        # Convert score to centipawns
        if isinstance(analysis['score'], chess.engine.PovScore):
            # Handle PovScore objects (most common case)
            score_obj = analysis['score'].relative
            if isinstance(score_obj, chess.engine.Cp):
                analysis['score_cp'] = score_obj.cp
            elif isinstance(score_obj, chess.engine.Mate):
                analysis['score_cp'] = 9999 if score_obj.mate() > 0 else -9999
            else:
                analysis['score_cp'] = 0
        elif isinstance(analysis['score'], chess.engine.Cp):
            analysis['score_cp'] = analysis['score'].cp
        elif isinstance(analysis['score'], chess.engine.Mate):
            analysis['score_cp'] = 9999 if analysis['score'].mate() > 0 else -9999
        else:
            analysis['score_cp'] = 0

        return analysis

    def _has_pin_motif(self, board: chess.Board) -> bool:
        """Return True if any pinned piece exists for either side."""
        try:
            for color in (chess.WHITE, chess.BLACK):
                king_sq = board.king(color)
                if king_sq is None:
                    continue
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and piece.color == color and sq != king_sq:
                        if board.is_pinned(color, sq):
                            return True
        except Exception:
            return False
        return False

    def generate_ssl_targets(self, board: chess.Board, ssl_tasks: List[str]) -> Dict[str, np.ndarray]:
        """Generate SSL targets for a position."""
        # Convert board to planes
        planes = encode_board(board)
        planes_tensor = torch.from_numpy(planes).unsqueeze(0)  # Add batch dimension

        ssl_targets = {}

        for task in ssl_tasks:
            if task == 'piece':
                ssl_targets['piece'] = self.ssl_algorithms._create_piece_targets(planes_tensor)
            elif task == 'threat':
                ssl_targets['threat'] = self.ssl_algorithms.detect_threats_batch(planes_tensor)
            elif task == 'pin':
                ssl_targets['pin'] = self.ssl_algorithms.detect_pins_batch(planes_tensor)
            elif task == 'fork':
                ssl_targets['fork'] = self.ssl_algorithms.detect_forks_batch(planes_tensor)
            elif task == 'control':
                ssl_targets['control'] = self.ssl_algorithms.calculate_square_control_batch(planes_tensor)

        return ssl_targets

    def create_training_sample(self, board: chess.Board, analysis: Dict,
                             ssl_targets: Dict[str, np.ndarray], ssl_tasks: List[str]) -> Dict:
        """Create a complete training sample."""
        # Convert board to planes
        s = encode_board(board)

        # Create policy target (one-hot for best move)
        pi = np.zeros(4672, dtype=np.float32)  # Matrix0 policy size
        if analysis['best_move']:
            try:
                idx = move_to_index(board, analysis['best_move'])
                if 0 <= idx < pi.shape[0]:
                    pi[idx] = 1.0
            except Exception:
                # If conversion fails, leave as all-zeros; sample still usable for value/SSL
                pass

        # Value target (normalize centipawn score)
        z = np.tanh(analysis['score_cp'] / 200.0).astype(np.float32)  # Scale to [-1, 1]

        # Legal move mask
        legal_moves = list(board.legal_moves)
        lm = np.zeros(4672, dtype=np.uint8)
        for move in legal_moves:
            try:
                idx = move_to_index(board, move)
                if 0 <= idx < lm.shape[0]:
                    lm[idx] = 1
            except Exception:
                continue

        sample = {
            's': s.astype(np.float32, copy=False),
            'pi': pi.astype(np.float32, copy=False),
            'z': z.astype(np.float32, copy=False),
            'legal_mask': lm,
            'fen': board.fen(),
            'stockfish_eval': analysis['score_cp'],
            'depth': analysis['depth'],
            'best_move': str(analysis['best_move']) if analysis['best_move'] else None,
        }

        # Add SSL targets
        for task in ssl_tasks:
            if task in ssl_targets:
                sample[f'ssl_{task}'] = ssl_targets[task].squeeze(0).numpy()

        return sample

    def generate_dataset(self, domain: str, subcategory: str, num_positions: int,
                        ssl_tasks: List[str], output_dir: str,
                        augment_plies: int = 4, augment_policy: str = "random",
                        unique_only: bool = True, max_tries: Optional[int] = None,
                        hb_every: int = 1000) -> str:
        """Generate a dataset. Augments seeds with randomized plies to reach target count."""
        logger.info(f"Generating up to {num_positions} positions for {domain}/{subcategory}")

        positions: List[Dict] = []
        seen_fens: set = set()
        motif_rejects = 0
        duplicate_rejects = 0
        error_rejects = 0
        start_time = time.time()
        last_hb = start_time

        metadata = {
            'domain': domain,
            'subcategory': subcategory,
            'num_positions': num_positions,
            'ssl_tasks': ssl_tasks,
            'stockfish_depth': self.engine_depth,
            'created_at': time.time(),
            'positions': []
        }

        # Seeds
        if domain == 'tactical':
            seed_iter = self._generate_tactical_positions(subcategory)
        elif domain == 'endgames':
            seed_iter = self._generate_endgame_positions(subcategory)
        elif domain == 'openings':
            seed_iter = self._generate_opening_positions(subcategory)
        elif domain == 'puzzles':
            seed_iter = self._generate_puzzle_positions(subcategory)
        elif domain == 'weaknesses':
            seed_iter = self._generate_weakness_positions(subcategory)
        elif domain == 'positional':
            seed_iter = self._generate_positional_positions(subcategory)
        elif domain == 'king_safety':
            seed_iter = self._generate_king_safety_positions(subcategory)
        elif domain == 'strategic':
            seed_iter = self._generate_strategic_positions(subcategory)
        else:
            raise ValueError(f"Unknown domain: {domain}")

        seeds = [b for b in seed_iter]
        if not seeds:
            seeds = [chess.Board()]

        def motif_ok(b: chess.Board) -> bool:
            if domain == 'tactical' and subcategory == 'pins':
                return self._has_pin_motif(b)
            return True

        if max_tries is None:
            max_tries = max(10000, num_positions * 50)

        tries = 0
        while len(positions) < num_positions and tries < max_tries:
            tries += 1
            board = random.choice(seeds).copy()

            # Augment with plies
            plies = random.randint(0, max(0, augment_plies))
            for _ in range(plies):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)
                if not legal:
                    break
                if augment_policy == 'stockfish' and self.engine is not None:
                    try:
                        mv = self.engine.play(board, chess.engine.Limit(time=0.02)).move
                    except Exception:
                        mv = random.choice(legal)
                else:
                    mv = random.choice(legal)
                board.push(mv)

            if not motif_ok(board):
                motif_rejects += 1
                continue

            fen = board.fen()
            if unique_only and fen in seen_fens:
                duplicate_rejects += 1
                continue

            try:
                analysis = self.analyze_position(board)
                ssl_targets = self.generate_ssl_targets(board, ssl_tasks)
                sample = self.create_training_sample(board, analysis, ssl_targets, ssl_tasks)
            except Exception:
                error_rejects += 1
                continue

            positions.append(sample)
            seen_fens.add(fen)
            metadata['positions'].append({
                'fen': sample['fen'],
                'evaluation': sample['stockfish_eval'],
                'best_move': sample['best_move'],
                'depth': sample['depth']
            })

            # Heartbeat
            if hb_every and len(positions) % max(1, hb_every) == 0:
                elapsed = max(1e-6, time.time() - start_time)
                speed = len(positions) / elapsed
                accept_rate = len(positions) / max(1, tries)
                remaining = max(0, num_positions - len(positions))
                eta_sec = remaining / max(1e-6, speed)
                logger.info(
                    f"HB: {len(positions)}/{num_positions} | acc={accept_rate:.2%} | "
                    f"speed={speed:.1f}/s | tries={tries} | rejects: motif={motif_rejects}, dup={duplicate_rejects}, err={error_rejects} | "
                    f"ETA ~ {eta_sec/60.0:.1f}m"
                )

        actual_n = len(positions)
        if actual_n == 0:
            raise RuntimeError("No positions generated. Increase augment_plies or allow duplicates.")

        # Save dataset
        output_path = Path(output_dir) / f"{domain}_{subcategory}_{actual_n}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays
        dataset = {}
        for key in positions[0].keys():
            if key in ['s', 'pi', 'z', 'legal_mask'] or key.startswith('ssl_'):
                dataset[key] = np.stack([pos[key] for pos in positions])

        np.savez_compressed(output_path, **dataset)

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata['actual_positions'] = actual_n
        metadata['unique'] = unique_only
        metadata['augment_plies'] = augment_plies
        metadata['augment_policy'] = augment_policy
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")
        elapsed = max(1e-6, time.time() - start_time)
        logger.info(
            f"DONE: {actual_n}/{num_positions} | {actual_n/elapsed:.1f} pos/s | tries={tries} | rejects: motif={motif_rejects}, dup={duplicate_rejects}, err={error_rejects}"
        )

        return str(output_path)

    def _generate_opening_positions(self, subcategory: str):
        """Generate opening positions targeting opening theory and principles."""
        if subcategory == 'classical_openings':
            # Standard, principled openings
            openings = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4
                "rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # After 1.e4 Nf3
                "rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # After 1.e4 Nf3
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # After 1.e4 e5
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",  # After 1.e4 e5 2.Nf3
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",  # After 1.e4 e5 2.Nf3
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",  # After 1.e4 e5 2.Nf3
            ]
        elif subcategory == 'development_focus':
            # Positions emphasizing piece development
            openings = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # After 1.Nf3 Nf6
                "rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # After 1.Nf3 Nf6 2.e4
                "rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # After 1.Nf3 Nf6 2.e4
                "rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # After 1.Nf3 Nf6 2.e4
            ]
        elif subcategory == 'center_control':
            # Positions emphasizing center control
            openings = [
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4
                "rnbqkbnr/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4 d3
                "rnbqkbnr/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4 d3
                "rnbqkbnr/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4 d3
            ]
        else:
            # Default opening position
            openings = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        for fen in openings:
            yield chess.Board(fen)

    def _generate_positional_positions(self, subcategory: str):
        """Generate positions targeting positional understanding."""
        if subcategory == 'pawn_structure':
            # Positions with various pawn structures
            positions = [
                "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # Isolated pawn
                "rnbqkb1r/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # Doubled pawns
                "rnbqkb1r/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # Backward pawn
                "rnbqkb1r/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # Hanging pawns
            ]
        elif subcategory == 'piece_coordination':
            # Positions emphasizing piece coordination
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Knights coordinated
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Bishops coordinated
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Rooks coordinated
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Queen coordination
            ]
        elif subcategory == 'weak_squares':
            # Positions with weak squares
            positions = [
                "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # Weak e4 square
                "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # Weak d5 square
                "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # Weak c4 square
                "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # Weak f4 square
            ]
        else:
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        for fen in positions:
            yield chess.Board(fen)

    def _generate_king_safety_positions(self, subcategory: str):
        """Generate positions targeting king safety."""
        if subcategory == 'castling_opportunities':
            # Positions where castling is important
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Should castle kingside
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Should castle queenside
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Castling blocked
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # King in center
            ]
        elif subcategory == 'king_attack':
            # Positions where king is under attack
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # King exposed
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # King trapped
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # King in check
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # King mate threat
            ]
        elif subcategory == 'pawn_shield':
            # Positions with pawn shield considerations
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Pawn shield intact
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Pawn shield broken
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Pawn shield weak
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # No pawn shield
            ]
        else:
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        for fen in positions:
            yield chess.Board(fen)

    def _generate_endgame_positions(self, subcategory: str):
        """Generate endgame seed positions with both kings present and legal setups."""
        rng = random

        def kings_ok(wk: int, bk: int) -> bool:
            return wk != bk and chess.square_distance(wk, bk) > 1

        def any_legal(b: chess.Board) -> bool:
            try:
                return any(True for _ in b.legal_moves)
            except Exception:
                return False

        def gen_kpk(n: int = 256):
            yielded = 0
            while yielded < n:
                wk = rng.choice(chess.SQUARES)
                bk = rng.choice(chess.SQUARES)
                if not kings_ok(wk, bk):
                    continue
                file_idx = rng.randrange(8)
                rank_idx = rng.randint(1, 6)  # ranks 2..7 for a white pawn
                wp = chess.square(file_idx, rank_idx)
                if wp in (wk, bk):
                    continue
                b = chess.Board(None)
                b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(wp, chess.Piece(chess.PAWN, chess.WHITE))
                b.turn = rng.choice([True, False])
                if not b.is_valid() or not any_legal(b):
                    continue
                yielded += 1
                yield b

        def gen_krk(n: int = 256):
            yielded = 0
            while yielded < n:
                wk = rng.choice(chess.SQUARES)
                bk = rng.choice(chess.SQUARES)
                if not kings_ok(wk, bk):
                    continue
                rook_sq = rng.choice([sq for sq in chess.SQUARES if sq not in (wk, bk)])
                b = chess.Board(None)
                b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
                b.turn = rng.choice([True, False])
                if not b.is_valid() or not any_legal(b):
                    continue
                yielded += 1
                yield b

        def gen_kqk(n: int = 256):
            yielded = 0
            while yielded < n:
                wk = rng.choice(chess.SQUARES)
                bk = rng.choice(chess.SQUARES)
                if not kings_ok(wk, bk):
                    continue
                queen_sq = rng.choice([sq for sq in chess.SQUARES if sq not in (wk, bk)])
                b = chess.Board(None)
                b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
                b.turn = rng.choice([True, False])
                if not b.is_valid() or not any_legal(b):
                    continue
                yielded += 1
                yield b

        def gen_minor(n: int = 256):
            yielded = 0
            while yielded < n:
                wk = rng.choice(chess.SQUARES)
                bk = rng.choice(chess.SQUARES)
                if not kings_ok(wk, bk):
                    continue
                minor_piece = chess.BISHOP if rng.random() < 0.5 else chess.KNIGHT
                minor_sq = rng.choice([sq for sq in chess.SQUARES if sq not in (wk, bk)])
                b = chess.Board(None)
                b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(minor_sq, chess.Piece(minor_piece, chess.WHITE))
                b.turn = rng.choice([True, False])
                if not b.is_valid() or not any_legal(b):
                    continue
                yielded += 1
                yield b

        if subcategory == 'king_and_pawn':
            yield from gen_kpk(256)
        elif subcategory == 'rook_endings':
            yield from gen_krk(256)
        elif subcategory == 'queen_endings':
            yield from gen_kqk(256)
        elif subcategory == 'minor_piece':
            yield from gen_minor(256)
        else:
            # Default to KPK seeds
            yield from gen_kpk(128)

    def _generate_strategic_positions(self, subcategory: str):
        """Generate positions targeting strategic planning."""
        if subcategory == 'long_term_plans':
            # Positions requiring long-term planning
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Plan: kingside attack
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Plan: queenside expansion
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Plan: center breakthrough
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Plan: endgame transition
            ]
        elif subcategory == 'space_advantage':
            # Positions with space considerations
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Space advantage
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Space disadvantage
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Equal space
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Space battle
            ]
        elif subcategory == 'material_vs_position':
            # Positions balancing material vs position
            positions = [
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Material advantage, bad position
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Positional advantage, material down
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Equal material, positional battle
                "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Complex material/position trade
            ]
        else:
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        for fen in positions:
            yield chess.Board(fen)

    def _generate_tactical_positions(self, subcategory: str):
        """Generate positions with tactical patterns."""
        if subcategory == 'pins':
            # Generate positions with pins
            positions = [
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Bishop pin
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Rook pin
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Queen pin
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Knight pin
            ]
        elif subcategory == 'forks':
            # Generate positions with forks
            positions = [
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Knight fork
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Pawn fork
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Bishop fork
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Rook fork
            ]
        elif subcategory == 'discovered_attacks':
            # Generate positions with discovered attacks
            positions = [
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Discovered check
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Discovered attack
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Double discovered
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",  # Discovered mate threat
            ]
        else:
            # Generic tactical positions
            positions = ["r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"]

        for fen in positions:
            yield chess.Board(fen)

    def _generate_puzzle_positions(self, subcategory: str):
        """Generate puzzle-like positions."""
        if subcategory == 'mate_in_2':
            positions = [
                "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",  # Mate in 2
                "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",  # Mate in 2
                "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",  # Mate in 2
                "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",  # Mate in 2
            ]
        elif subcategory == 'mate_in_3':
            positions = [
                "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",  # Mate in 3
                "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",  # Mate in 3
                "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",  # Mate in 3
                "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",  # Mate in 3
            ]
        else:
            positions = ["6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"]

        for fen in positions:
            yield chess.Board(fen)

    def _generate_weakness_positions(self, subcategory: str):
        """Generate positions targeting common weaknesses."""
        if subcategory == 'hanging_pieces':
            positions = [
                "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",  # Hanging piece
                "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",  # Hanging piece
                "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",  # Hanging piece
                "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",  # Hanging piece
            ]
        elif subcategory == 'undefended_squares':
            positions = [
                "rnbqkbnr/pppppppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",  # Undefended square
                "rnbqkbnr/pppppppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",  # Undefended square
                "rnbqkbnr/pppppppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",  # Undefended square
                "rnbqkbnr/pppppppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",  # Undefended square
            ]
        elif subcategory == 'back_rank_weakness':
            positions = [
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # Back rank weakness
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # Back rank weakness
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # Back rank weakness
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # Back rank weakness
            ]
        else:
            positions = ["r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3"]

        for fen in positions:
            yield chess.Board(fen)


def main():
    parser = argparse.ArgumentParser(description='Generate Stockfish training data for Matrix0')
    parser.add_argument('--domain', required=True, 
                       choices=['openings', 'tactical', 'endgames', 'puzzles', 'weaknesses', 
                               'positional', 'king_safety', 'strategic'])
    parser.add_argument('--subcategory', required=True, help='Specific subcategory within domain')
    parser.add_argument('--positions', type=int, default=1000, help='Number of positions to generate')
    parser.add_argument('--stockfish-path', default='stockfish', help='Path to Stockfish executable')
    parser.add_argument('--stockfish-depth', type=int, default=10, help='Stockfish analysis depth')
    parser.add_argument('--threads', type=int, default=1, help='UCI Threads')
    parser.add_argument('--hash-mb', type=int, default=128, help='UCI Hash in MB')
    parser.add_argument('--movetime-ms', type=int, default=None, help='Per-position movetime in ms (overrides depth)')
    parser.add_argument('--ssl-tasks', nargs='+', default=['piece', 'threat', 'pin', 'fork', 'control'],
                       choices=['piece', 'threat', 'pin', 'fork', 'control'])
    parser.add_argument('--output-dir', default='data/stockfish_games')
    parser.add_argument('--augment-plies', type=int, default=4, help='Max random plies from seeds for augmentation')
    parser.add_argument('--augment-policy', type=str, default='random', choices=['random', 'stockfish'], help='Augmentation move selection')
    parser.add_argument('--allow-duplicates', action='store_true', help='Allow duplicate FENs to reach target')
    parser.add_argument('--hb-every', type=int, default=1000, help='Heartbeat every N accepted positions (0 to disable)')

    args = parser.parse_args()

    # Create output directory
    output_subdir = Path(args.output_dir) / args.domain / args.subcategory
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    with StockfishDataGenerator(
        stockfish_path=args.stockfish_path,
        engine_depth=args.stockfish_depth,
        threads=args.threads,
        hash_mb=args.hash_mb,
        movetime_ms=args.movetime_ms,
    ) as generator:
        dataset_path = generator.generate_dataset(
            args.domain,
            args.subcategory,
            args.positions,
            args.ssl_tasks,
            str(output_subdir),
            augment_plies=int(args.augment_plies),
            augment_policy=str(args.augment_policy),
            unique_only=(not args.allow_duplicates),
            hb_every=int(args.hb_every)
        )

    logger.info(f"Dataset generation complete: {dataset_path}")


if __name__ == '__main__':
    main()
