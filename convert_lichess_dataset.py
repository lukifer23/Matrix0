#!/usr/bin/env python3
"""
Lichess Dataset Converter for Matrix0
Converts CSV chess games to our training format (.npz files)
"""

import csv
import chess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging

# Import our encoding functions
import sys
sys.path.append('.')
from azchess.encoding import encode_board, move_to_index

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LichessGameConverter:
    def __init__(self, quality_filters: Dict = None):
        self.quality_filters = quality_filters or {
            'min_rating': 1500,      # Minimum player rating
            'max_rating': 3000,      # Maximum player rating
            'min_turns': 20,         # Minimum game length
            'max_turns': 150,        # Maximum game length
            'min_avg_rating': 1600,  # Minimum average rating
        }
        
    def parse_moves(self, moves_str: str) -> List[str]:
        """Parse Standard Algebraic Notation (SAN) move string into list of moves"""
        if not moves_str or moves_str.strip() == '':
            return []
        return moves_str.strip().split()
    
    def validate_game(self, row: Dict) -> bool:
        """Check if game meets quality criteria"""
        try:
            # Rating checks
            white_rating = int(row['white_rating'])
            black_rating = int(row['black_rating'])
            avg_rating = (white_rating + black_rating) / 2
            
            if (white_rating < self.quality_filters['min_rating'] or 
                black_rating < self.quality_filters['min_rating'] or
                white_rating > self.quality_filters['max_rating'] or
                black_rating > self.quality_filters['max_rating'] or
                avg_rating < self.quality_filters['min_avg_rating']):
                return False
            
            # Game length checks
            turns = int(row['turns'])
            if turns < self.quality_filters['min_turns'] or turns > self.quality_filters['max_turns']:
                return False
            
            # Victory status checks (exclude incomplete games)
            victory_status = row['victory_status']
            if victory_status in ['abandoned', 'timeout']:
                return False
                
            return True
            
        except (ValueError, KeyError):
            return False
    
    def convert_game_to_training_format(self, row: Dict) -> Optional[Dict]:
        """Convert a single game to our training format"""
        try:
            # Parse moves (these are in SAN notation, not UCI)
            moves_str = row['moves']
            moves = self.parse_moves(moves_str)
            if len(moves) < 10:  # Skip very short games
                return None
            
            # Create chess board and play through the game
            board = chess.Board()
            states = []
            pis = []
            turns = []
            
            # Determine game result
            winner = row['winner']
            victory_status = row['victory_status']
            
            if victory_status == 'mate':
                if winner == 'white':
                    final_result = 1.0
                elif winner == 'black':
                    final_result = -1.0
                else:
                    final_result = 0.0
            elif victory_status == 'resign':
                if winner == 'white':
                    final_result = 1.0
                elif winner == 'black':
                    final_result = -1.0
                else:
                    final_result = 0.0
            elif victory_status == 'draw':
                final_result = 0.0
            else:  # outoftime, etc.
                if winner == 'white':
                    final_result = 1.0
                elif winner == 'black':
                    final_result = -1.0
                else:
                    final_result = 0.0
            
            # Play through each move
            for i, move_san in enumerate(moves):
                if board.is_game_over():
                    break
                    
                # Encode current position
                state = encode_board(board)
                states.append(state)
                
                # Parse SAN move to UCI move
                try:
                    # Parse SAN move (e.g., "d4", "Nc6", "O-O")
                    move = board.parse_san(move_san)
                    
                    # Create policy (one-hot for the actual move played)
                    pi = np.zeros(4672, dtype=np.float32)
                    move_idx = move_to_index(board, move)
                    pi[move_idx] = 1.0
                    
                    pis.append(pi)
                    turns.append(1 if board.turn == chess.WHITE else -1)
                    
                    # Make the move
                    board.push(move)
                    
                except (ValueError, chess.InvalidMoveError) as e:
                    # Skip invalid moves
                    logger.debug(f"Invalid move '{move_san}' in game {row.get('id', 'unknown')}: {e}")
                    continue
            
            if len(states) < 10:  # Skip games with too few valid positions
                return None
            
            # Ensure all arrays have the same length
            if len(states) != len(pis) or len(states) != len(turns):
                logger.warning(f"Array length mismatch: states={len(states)}, pis={len(pis)}, turns={len(turns)}")
                return None
            
            # Create training data
            game_data = {
                's': np.array(states, dtype=np.float32),
                'pi': np.array(pis, dtype=np.float32),
                'z': np.array([final_result * t for t in turns], dtype=np.float32),
                'metadata': {
                    'game_id': row['id'],
                    'white_rating': int(row['white_rating']),
                    'black_rating': int(row['black_rating']),
                    'opening_name': row['opening_name'],
                    'victory_status': row['victory_status'],
                    'winner': row['winner'],
                    'turns': len(states),
                    'source': 'lichess'
                }
            }
            
            # Validate the data before returning
            if game_data['s'].shape[0] == 0 or game_data['pi'].shape[0] == 0 or game_data['z'].shape[0] == 0:
                logger.warning(f"Empty arrays in game {row.get('id', 'unknown')}")
                return None
                
            return game_data
            
        except Exception as e:
            logger.warning(f"Failed to convert game {row.get('id', 'unknown')}: {e}")
            return None
    
    def convert_dataset(self, csv_path: str, output_dir: str, max_games: int = None) -> int:
        """Convert the entire dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        skipped_count = 0
        
        logger.info(f"Starting conversion of {csv_path}")
        logger.info(f"Quality filters: {self.quality_filters}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_games = sum(1 for _ in open(csv_path)) - 1  # Subtract header
            
            for i, row in enumerate(tqdm(reader, total=total_games, desc="Converting games")):
                if max_games and converted_count >= max_games:
                    break
                
                # Validate game quality
                if not self.validate_game(row):
                    skipped_count += 1
                    continue
                
                # Convert game
                game_data = self.convert_game_to_training_format(row)
                if game_data is None:
                    skipped_count += 1
                    continue
                
                # Save game
                game_id = row['id']
                filename = f"lichess_{game_id}.npz"
                filepath = output_path / filename
                
                # Save without metadata (npz doesn't handle dicts well)
                np.savez_compressed(
                    filepath,
                    s=game_data['s'],
                    pi=game_data['pi'],
                    z=game_data['z']
                )
                
                converted_count += 1
                
                if converted_count % 100 == 0:
                    logger.info(f"Converted {converted_count} games, skipped {skipped_count}")
        
        logger.info(f"Conversion complete!")
        logger.info(f"Converted: {converted_count} games")
        logger.info(f"Skipped: {skipped_count} games")
        logger.info(f"Output directory: {output_path}")
        
        return converted_count

def main():
    parser = argparse.ArgumentParser(description="Convert Lichess dataset to Matrix0 training format")
    parser.add_argument("--input", type=str, default="games.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/lichess", help="Output directory")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum games to convert")
    parser.add_argument("--min-rating", type=int, default=1500, help="Minimum player rating")
    parser.add_argument("--max-rating", type=int, default=3000, help="Maximum player rating")
    parser.add_argument("--min-turns", type=int, default=20, help="Minimum game length")
    parser.add_argument("--max-turns", type=int, default=150, help="Maximum game length")
    
    args = parser.parse_args()
    
    # Setup quality filters
    quality_filters = {
        'min_rating': args.min_rating,
        'max_rating': args.max_rating,
        'min_turns': args.min_turns,
        'max_turns': args.max_turns,
        'min_avg_rating': args.min_rating + 100,  # Average rating should be higher
    }
    
    # Create converter and run
    converter = LichessGameConverter(quality_filters)
    converted_count = converter.convert_dataset(
        args.input, 
        args.output, 
        max_games=args.max_games
    )
    
    print(f"\n🎯 Successfully converted {converted_count} high-quality games!")
    print(f"📁 Output saved to: {args.output}")
    print(f"🔧 Quality filters applied: {quality_filters}")

if __name__ == "__main__":
    main()
