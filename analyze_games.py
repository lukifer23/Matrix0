#!/usr/bin/env python3
import numpy as np
import glob
from datetime import datetime

# Find all games from today
games = sorted(glob.glob('data/selfplay/selfplay_w*_g*_2025-08-16T15:*.npz'))

print("=== SELF-PLAY ANALYSIS ===")
print(f"Total games: {len(games)}")

total_moves = 0
results = []
game_details = []

for game_file in games:
    try:
        data = np.load(game_file)
        moves = len(data['s'])
        result = np.unique(data['z'])
        total_moves += moves
        
        # Extract timestamp from filename
        timestamp = game_file.split('_')[-1].replace('.npz', '')
        game_details.append((timestamp, moves, result))
        
        results.extend(result)
        
    except Exception as e:
        print(f"Error loading {game_file}: {e}")

print(f"Total moves: {total_moves}")
print(f"Average moves per game: {total_moves/len(games):.1f}")

# Analyze results
unique_results, counts = np.unique(results, return_counts=True)
print(f"Results: {dict(zip(unique_results, counts))}")

# Calculate win rates
white_wins = np.sum(np.array(results) > 0)
black_wins = np.sum(np.array(results) < 0)
draws = np.sum(np.array(results) == 0)

print(f"White wins: {white_wins} ({white_wins/len(results):.1%})")
print(f"Black wins: {black_wins} ({black_wins/len(results):.1%})")
print(f"Draws: {draws} ({draws/len(results):.1%})")

print("\n=== GAME DETAILS ===")
for timestamp, moves, result in game_details:
    print(f"{timestamp}: {moves} moves, result: {result}")

# Check current config
print("\n=== CURRENT CONFIG ===")
print("Check config.yaml for actual simulation counts being used")
