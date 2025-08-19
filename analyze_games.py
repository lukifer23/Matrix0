#!/usr/bin/env python3
"""Analyze self-play games."""

import argparse
import glob
import logging
import os

import numpy as np


def analyze_games(directory: str, date_pattern: str) -> None:
    """Summarize self-play game statistics.

    Parameters
    ----------
    directory: str
        Directory containing self-play game files.
    date_pattern: str
        Glob pattern representing the date/time portion of filenames.
    """
    games = sorted(
        glob.glob(os.path.join(directory, f"selfplay_w*_g*_{date_pattern}.npz"))
    )

    logging.info("=== SELF-PLAY ANALYSIS ===")
    logging.info("Total games: %s", len(games))

    total_moves = 0
    results = []
    game_details = []

    for game_file in games:
        try:
            data = np.load(game_file)
            moves = len(data["s"])
            result = np.unique(data["z"])
            total_moves += moves

            # Extract timestamp from filename
            timestamp = game_file.split("_")[-1].replace(".npz", "")
            game_details.append((timestamp, moves, result))

            results.extend(result)

        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Error loading %s: %s", game_file, exc)

    if not games:
        logging.info("No games found.")
        return

    logging.info("Total moves: %s", total_moves)
    logging.info("Average moves per game: %.1f", total_moves / len(games))

    # Analyze results
    unique_results, counts = np.unique(results, return_counts=True)
    logging.info("Results: %s", dict(zip(unique_results, counts)))

    # Calculate win rates
    white_wins = np.sum(np.array(results) > 0)
    black_wins = np.sum(np.array(results) < 0)
    draws = np.sum(np.array(results) == 0)

    logging.info("White wins: %s (%s)", white_wins, f"{white_wins/len(results):.1%}")
    logging.info("Black wins: %s (%s)", black_wins, f"{black_wins/len(results):.1%}")
    logging.info("Draws: %s (%s)", draws, f"{draws/len(results):.1%}")

    logging.info("\n=== GAME DETAILS ===")
    for timestamp, moves, result in game_details:
        logging.info("%s: %s moves, result: %s", timestamp, moves, result)

    logging.info("\n=== CURRENT CONFIG ===")
    logging.info("Check config.yaml for actual simulation counts being used")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze self-play games")
    parser.add_argument(
        "--directory",
        default="data/selfplay",
        help="Directory containing self-play game files",
    )
    parser.add_argument(
        "--date-pattern",
        default="2025-08-16T15:*",
        help="Date pattern to match in filenames (e.g., '2025-08-16T15:*')",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyze_games(args.directory, args.date_pattern)


if __name__ == "__main__":
    main()
