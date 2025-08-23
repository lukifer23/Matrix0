#!/usr/bin/env python3
"""
Lichess Data Processing Tool for Matrix0

This script provides functionalities to download, filter, and convert
Lichess game data into the .npz format compatible with Matrix0 training.

Workflow:
1. Download: Fetch monthly PGN archives from the Lichess database.
2. Filter: Parse the PGNs and filter games based on criteria like player Elo,
   time control, and game termination status. This step produces a clean PGN.
3. Convert: Read the filtered PGN and convert the game states into the .npz
   format (state, policy, value) required by the training pipeline.
"""

import argparse
import bz2
import logging
import os
from pathlib import Path
import requests
from tqdm import tqdm
import chess.pgn
import numpy as np

from azchess.encoding import encode_board, move_to_index
from azchess.logging_utils import setup_logging

# Setup logging
logger = setup_logging(level=logging.INFO)

# --- Configuration ---
LICHESS_DB_URL = "https://database.lichess.org/standard/"
DEFAULT_OUTPUT_DIR = Path("data/lichess")
DEFAULT_PGN_DIR = Path("data/lichess/pgn")

# --- Helper Functions ---

def download_file(url: str, dest: Path):
    """Download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {dest.name}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.info(f"Successfully downloaded {dest.name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink() # Clean up partial download

def process_game(game: chess.pgn.Game, min_elo: int):
    """Extracts training samples from a single game if it meets criteria."""
    headers = game.headers
    
    # 1. Filter based on headers
    if headers.get("Termination") != "Normal":
        return None
    if "FICSGames" in headers.get("Site", ""): # Often low quality
        return None
        
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        if white_elo < min_elo or black_elo < min_elo:
            return None
    except ValueError:
        return None

    # 2. Determine game result
    result = headers.get("Result", "*")
    if result == "1-0":
        z = 1.0
    elif result == "0-1":
        z = -1.0
    elif result == "1/2-1/2":
        z = 0.0
    else:
        return None # Skip games with unknown results

    # 3. Process moves
    board = game.board()
    samples = []
    for move in game.mainline_moves():
        s = encode_board(board)
        pi = np.zeros(4672, dtype=np.float32)
        pi[move_to_index(board, move)] = 1.0
        
        # Value is from the perspective of the current player
        value = z if board.turn == chess.WHITE else -z
        samples.append({"s": s, "pi": pi, "z": np.float32(value)})
        
        board.push(move)
        
    return samples

# --- Main Functions ---

def download_lichess(month: str, pgn_dir: Path = DEFAULT_PGN_DIR):
    """
    Downloads a monthly PGN archive from Lichess.
    Example month: "2023-01"
    """
    filename = f"lichess_db_standard_rated_{month}.pgn.bz2"
    url = f"{LICHESS_DB_URL}{filename}"
    dest_path = pgn_dir / filename
    
    if dest_path.exists():
        logger.info(f"{dest_path.name} already exists. Skipping download.")
        return
        
    download_file(url, dest_path)

def convert_pgn_to_npz(
    pgn_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_elo: int = 2000,
    max_games: int = 100000
):
    """
    Filters and converts a PGN file to NPZ shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing PGN: {pgn_path.name}")
    logger.info(f"Filter criteria: min_elo={min_elo}, max_games={max_games}")

    opener = bz2.open if pgn_path.suffix == ".bz2" else open
    
    game_count = 0
    shard_count = 0
    samples_in_shard = []
    SHARD_SIZE = 8192 # Number of samples per .npz file

    with opener(pgn_path, 'rt') as pgn_file:
        pbar = tqdm(desc="Processing games", unit=" games")
        while game_count < max_games:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                pbar.update(1)
                game_samples = process_game(game, min_elo)
                
                if game_samples:
                    game_count += 1
                    samples_in_shard.extend(game_samples)
                    
                    if len(samples_in_shard) >= SHARD_SIZE:
                        shard_name = f"lichess_{os.urandom(4).hex()}.npz"
                        save_path = output_dir / shard_name
                        
                        # Collate samples
                        s_batch = np.stack([s['s'] for s in samples_in_shard])
                        pi_batch = np.stack([s['pi'] for s in samples_in_shard])
                        z_batch = np.stack([s['z'] for s in samples_in_shard])
                        
                        np.savez_compressed(save_path, s=s_batch, pi=pi_batch, z=z_batch)
                        shard_count += 1
                        logger.info(f"Saved shard {shard_name} with {len(samples_in_shard)} samples.")
                        samples_in_shard = []

            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed game: {e}")
                continue
        pbar.close()

    # Save any remaining samples in the last shard
    if samples_in_shard:
        shard_name = f"lichess_{os.urandom(4).hex()}.npz"
        save_path = output_dir / shard_name
        s_batch = np.stack([s['s'] for s in samples_in_shard])
        pi_batch = np.stack([s['pi'] for s in samples_in_shard])
        z_batch = np.stack([s['z'] for s in samples_in_shard])
        np.savez_compressed(save_path, s=s_batch, pi=pi_batch, z=z_batch)
        shard_count += 1
        logger.info(f"Saved final shard {shard_name} with {len(samples_in_shard)} samples.")

    logger.info(f"Conversion complete. Processed {game_count} games and created {shard_count} shards.")

def main():
    parser = argparse.ArgumentParser(description="Matrix0 Lichess Data Processor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    parser_download = subparsers.add_parser("download", help="Download PGN archives from Lichess.")
    parser_download.add_argument("month", type=str, help="Month to download in YYYY-MM format.")
    parser_download.add_argument("--pgn-dir", type=Path, default=DEFAULT_PGN_DIR, help="Directory to save PGN files.")

    # Convert command
    parser_convert = subparsers.add_parser("convert", help="Convert a PGN file to NPZ format.")
    parser_convert.add_argument("pgn_file", type=Path, help="Path to the PGN file (can be .pgn or .pgn.bz2).")
    parser_convert.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save NPZ shards.")
    parser_convert.add_argument("--min-elo", type=int, default=2200, help="Minimum Elo of both players.")
    parser_convert.add_argument("--max-games", type=int, default=100000, help="Maximum number of games to process from the PGN.")

    args = parser.parse_args()

    if args.command == "download":
        download_lichess(args.month, args.pgn_dir)
    elif args.command == "convert":
        if not args.pgn_file.exists():
            logger.error(f"PGN file not found: {args.pgn_file}")
            return
        convert_pgn_to_npz(args.pgn_file, args.output_dir, args.min_elo, args.max_games)

if __name__ == "__main__":
    main()
