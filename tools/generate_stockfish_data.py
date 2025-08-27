#!/usr/bin/env python3
"""
Stockfish Data Generator for Matrix0 Training

Generates high-quality training datasets using Stockfish analysis with SSL annotations.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch

from azchess.ssl_algorithms import ChessSSLAlgorithms
from azchess.encoding import board_to_planes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockfishDataGenerator:
    """Generates training data using Stockfish analysis with SSL annotations."""

    def __init__(self, stockfish_path: str = "stockfish", engine_depth: int = 10):
        self.stockfish_path = stockfish_path
        self.engine_depth = engine_depth
        self.ssl_algorithms = ChessSSLAlgorithms()
        self.engine = None

    def __enter__(self):
        """Start Stockfish engine."""
        logger.info(f"Starting Stockfish engine: {self.stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": 4, "Hash": 512})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close Stockfish engine."""
        if self.engine:
            logger.info("Closing Stockfish engine")
            self.engine.quit()

    def analyze_position(self, board: chess.Board, depth: Optional[int] = None) -> Dict:
        """Analyze a position with Stockfish."""
        if depth is None:
            depth = self.engine_depth

        logger.debug(f"Analyzing position at depth {depth}")

        # Get best move and evaluation
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))

        # Extract analysis data
        analysis = {
            'score': info.get('score', chess.engine.Cp(0)),
            'best_move': info.get('pv', [None])[0] if info.get('pv') else None,
            'depth': info.get('depth', 0),
            'nodes': info.get('nodes', 0),
            'time': info.get('time', 0),
        }

        # Convert score to centipawns
        if isinstance(analysis['score'], chess.engine.Cp):
            analysis['score_cp'] = analysis['score'].cp
        elif isinstance(analysis['score'], chess.engine.Mate):
            analysis['score_cp'] = 9999 if analysis['score'].mate() > 0 else -9999
        else:
            analysis['score_cp'] = 0

        return analysis

    def generate_ssl_targets(self, board: chess.Board, ssl_tasks: List[str]) -> Dict[str, np.ndarray]:
        """Generate SSL targets for a position."""
        # Convert board to planes
        planes = board_to_planes(board)
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
                ssl_targets['control'] = self.ssl_algorithms.detect_square_control_batch(planes_tensor)

        return ssl_targets

    def create_training_sample(self, board: chess.Board, analysis: Dict,
                             ssl_targets: Dict[str, np.ndarray], ssl_tasks: List[str]) -> Dict:
        """Create a complete training sample."""
        # Convert board to planes
        s = board_to_planes(board)

        # Create policy target (one-hot for best move)
        pi = np.zeros(4672)  # Matrix0 policy size
        if analysis['best_move']:
            move_idx = self._move_to_index(analysis['best_move'])
            if move_idx < len(pi):
                pi[move_idx] = 1.0

        # Value target (normalize centipawn score)
        z = np.tanh(analysis['score_cp'] / 200.0)  # Scale to [-1, 1]

        # Legal move mask
        legal_moves = list(board.legal_moves)
        lm = np.zeros(4672)
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if move_idx < len(lm):
                lm[move_idx] = 1.0

        sample = {
            's': s,
            'pi': pi,
            'z': z,
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

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to policy index (simplified implementation)."""
        # This is a placeholder - actual implementation would need proper move encoding
        # Matrix0 uses a specific move encoding scheme
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Simplified encoding - replace with actual Matrix0 encoding
        base_idx = from_sq * 64 + to_sq
        if promotion:
            promotion_offset = {'q': 0, 'r': 1, 'b': 2, 'n': 3}[promotion.symbol()]
            base_idx += (64 * 64) + (promotion_offset * 64 * 64)
        return base_idx

    def generate_dataset(self, domain: str, subcategory: str, num_positions: int,
                        ssl_tasks: List[str], output_dir: str) -> str:
        """Generate a complete dataset."""
        logger.info(f"Generating {num_positions} positions for {domain}/{subcategory}")

        positions = []
        metadata = {
            'domain': domain,
            'subcategory': subcategory,
            'num_positions': num_positions,
            'ssl_tasks': ssl_tasks,
            'stockfish_depth': self.engine_depth,
            'created_at': time.time(),
            'positions': []
        }

        # Generate positions based on domain
        if domain == 'tactical':
            board_generator = self._generate_tactical_positions(subcategory)
        elif domain == 'endgames':
            board_generator = self._generate_endgame_positions(subcategory)
        elif domain == 'openings':
            board_generator = self._generate_opening_positions(subcategory)
        else:
            raise ValueError(f"Unknown domain: {domain}")

        for i, board in enumerate(board_generator):
            if i >= num_positions:
                break

            logger.debug(f"Processing position {i+1}/{num_positions}")

            # Analyze with Stockfish
            analysis = self.analyze_position(board)

            # Generate SSL targets
            ssl_targets = self.generate_ssl_targets(board, ssl_tasks)

            # Create training sample
            sample = self.create_training_sample(board, analysis, ssl_targets, ssl_tasks)

            positions.append(sample)
            metadata['positions'].append({
                'fen': sample['fen'],
                'evaluation': sample['stockfish_eval'],
                'best_move': sample['best_move'],
                'depth': sample['depth']
            })

        # Save dataset
        output_path = Path(output_dir) / f"{domain}_{subcategory}_{num_positions}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays
        dataset = {}
        for key in positions[0].keys():
            if key in ['s', 'pi', 'z', 'legal_mask'] or key.startswith('ssl_'):
                dataset[key] = np.stack([pos[key] for pos in positions])

        # Save dataset
        np.savez_compressed(output_path, **dataset)

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")

        return str(output_path)

    def _generate_tactical_positions(self, subcategory: str):
        """Generate positions with tactical patterns."""
        # Placeholder - implement specific tactical generators
        # This would create positions with pins, forks, discoveries, etc.

        if subcategory == 'pins':
            # Generate positions with pins
            yield chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
        elif subcategory == 'forks':
            # Generate positions with forks
            yield chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
        else:
            # Generic tactical positions
            yield chess.Board()

    def _generate_endgame_positions(self, subcategory: str):
        """Generate endgame positions."""
        # Placeholder - implement endgame generators
        if subcategory == 'king_and_pawn':
            yield chess.Board("8/8/8/8/8/8/4K3/4P3 w - - 0 1")
        else:
            yield chess.Board("8/8/8/8/8/8/4K3/4Q3 w - - 0 1")

    def _generate_opening_positions(self, subcategory: str):
        """Generate opening positions."""
        # Placeholder - implement opening generators
        yield chess.Board()


def main():
    parser = argparse.ArgumentParser(description='Generate Stockfish training data for Matrix0')
    parser.add_argument('--domain', required=True, choices=['openings', 'tactical', 'endgames', 'puzzles', 'weaknesses'])
    parser.add_argument('--subcategory', required=True, help='Specific subcategory within domain')
    parser.add_argument('--positions', type=int, default=1000, help='Number of positions to generate')
    parser.add_argument('--stockfish-path', default='stockfish', help='Path to Stockfish executable')
    parser.add_argument('--stockfish-depth', type=int, default=10, help='Stockfish analysis depth')
    parser.add_argument('--ssl-tasks', nargs='+', default=['piece', 'threat', 'pin', 'fork', 'control'],
                       choices=['piece', 'threat', 'pin', 'fork', 'control'])
    parser.add_argument('--output-dir', default='data/stockfish_games')

    args = parser.parse_args()

    # Create output directory
    output_subdir = Path(args.output_dir) / args.domain / args.subcategory
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    with StockfishDataGenerator(args.stockfish_path, args.stockfish_depth) as generator:
        dataset_path = generator.generate_dataset(
            args.domain,
            args.subcategory,
            args.positions,
            args.ssl_tasks,
            str(output_subdir)
        )

    logger.info(f"Dataset generation complete: {dataset_path}")


if __name__ == '__main__':
    main()
