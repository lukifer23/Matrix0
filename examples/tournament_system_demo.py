#!/usr/bin/env python3
"""
Tournament System Demonstration
Showcases the advanced tournament management system for Matrix0.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.ratings import Glicko2Rating, update_glicko2_player
from benchmarks.tournament import (
    Tournament, TournamentConfig, TournamentFormat, TournamentResult,
    EngineStanding, GameResult, create_tournament_config, run_tournament
)


def demo_rating_system():
    """Demonstrate the Glicko-2 rating system."""
    print("=" * 60)
    print("Glicko-2 Rating System Demo")
    print("=" * 60)

    # Create initial ratings
    matrix0_rating = Glicko2Rating(rating=1500, rd=350, sigma=0.06)
    stockfish_rating = Glicko2Rating(rating=2000, rd=50, sigma=0.05)
    lc0_rating = Glicko2Rating(rating=1900, rd=75, sigma=0.055)

    print("Initial Ratings:")
    print(f"  Matrix0: {matrix0_rating.rating:.0f} ± {matrix0_rating.rd:.0f}")
    print(f"  Stockfish: {stockfish_rating.rating:.0f} ± {stockfish_rating.rd:.0f}")
    print(f"  LC0: {lc0_rating.rating:.0f} ± {lc0_rating.rd:.0f}")
    print()

    # Simulate tournament results
    print("Simulating tournament results...")

    # Matrix0 vs Stockfish: Matrix0 wins
    matrix0_rating, stockfish_rating = update_glicko2_player(
        matrix0_rating, stockfish_rating, 1.0  # Matrix0 wins
    )

    # Matrix0 vs LC0: Draw
    matrix0_rating, lc0_rating = update_glicko2_player(
        matrix0_rating, lc0_rating, 0.5  # Draw
    )

    # Stockfish vs LC0: Stockfish wins
    stockfish_rating, lc0_rating = update_glicko2_player(
        stockfish_rating, lc0_rating, 1.0  # Stockfish wins
    )

    print("\nUpdated Ratings after tournament:")
    print(f"  Matrix0: {matrix0_rating.rating:.0f} ± {matrix0_rating.rd:.0f}")
    print(f"  Stockfish: {stockfish_rating.rating:.0f} ± {stockfish_rating.rd:.0f}")
    print(f"  LC0: {lc0_rating.rating:.0f} ± {lc0_rating.rd:.0f}")

    print("\n✅ Glicko-2 rating system demonstration complete")


def demo_tournament_config():
    """Demonstrate tournament configuration."""
    print("\n" + "=" * 60)
    print("Tournament Configuration Demo")
    print("=" * 60)

    # Create tournament configurations for different formats
    configs = {
        "Round Robin": create_tournament_config(
            name="Round Robin Championship",
            engines=["Matrix0", "Stockfish", "LC0"],
            format=TournamentFormat.ROUND_ROBIN,
            num_games_per_pairing=4,
            time_control_ms=5000
        ),
        "Single Elimination": create_tournament_config(
            name="Knockout Tournament",
            engines=["Matrix0", "Stockfish", "LC0", "Komodo"],
            format=TournamentFormat.SINGLE_ELIMINATION,
            num_games_per_pairing=1,
            time_control_ms=10000
        ),
        "Swiss": create_tournament_config(
            name="Swiss Championship",
            engines=["Matrix0", "Stockfish", "LC0", "Komodo", "Houdini"],
            format=TournamentFormat.SWISS,
            num_games_per_pairing=2,
            time_control_ms=3000
        )
    }

    for format_name, config in configs.items():
        print(f"\n{format_name} Tournament:")
        print(f"  Name: {config.name}")
        print(f"  Engines: {', '.join(config.engines)}")
        print(f"  Format: {config.format.value}")
        print(f"  Games per pairing: {config.num_games_per_pairing}")
        print(f"  Time control: {config.time_control_ms}ms")
        print(f"  Max concurrency: {config.max_concurrency}")

    print("\n✅ Tournament configuration demonstration complete")


def demo_tournament_simulation():
    """Demonstrate tournament simulation."""
    print("\n" + "=" * 60)
    print("Tournament Simulation Demo")
    print("=" * 60)

    # Create a simple tournament
    config = create_tournament_config(
        name="Demo Tournament",
        engines=["Matrix0", "Stockfish"],
        format=TournamentFormat.ROUND_ROBIN,
        num_games_per_pairing=2,
        time_control_ms=1000
    )

    print(f"Creating tournament: {config.name}")
    print(f"Format: {config.format.value}")
    print(f"Participants: {', '.join(config.engines)}")

    # For demo purposes, we'll create a tournament object but not actually run it
    # since that would require engine executables
    tournament = Tournament(config)

    print(f"\nTournament created with {len(tournament.pairings)} pairings")
    print("Pairings:")
    for i, pairing in enumerate(tournament.pairings, 1):
        print(f"  Round {i}: {pairing[0]} vs {pairing[1]}")

    print("\n✅ Tournament simulation demonstration complete")


def demo_engine_standings():
    """Demonstrate tournament standings and results."""
    print("\n" + "=" * 60)
    print("Tournament Standings Demo")
    print("=" * 60)

    # Create sample standings
    standings = [
        EngineStanding(
            engine_name="Stockfish",
            games_played=10,
            wins=8,
            losses=1,
            draws=1,
            score=8.5,
            rating=1950
        ),
        EngineStanding(
            engine_name="Matrix0",
            games_played=10,
            wins=6,
            losses=2,
            draws=2,
            score=7.0,
            rating=1850
        ),
        EngineStanding(
            engine_name="LC0",
            games_played=10,
            wins=4,
            losses=4,
            draws=2,
            score=5.0,
            rating=1750
        )
    ]

    print("Tournament Standings:")
    print("Rank | Engine     | Score | W-L-D | Rating")
    print("-" * 45)

    for i, standing in enumerate(standings, 1):
        print("3d"
              "12"
              "6.1f"
              "3d"
              "4d")

    # Calculate performance metrics
    total_games = sum(s.games_played for s in standings)
    win_rate = sum(s.wins for s in standings) / total_games * 100

    print(f"\nTournament Statistics:")
    print(f"  Total games played: {total_games}")
    print(f"  Overall win rate: {win_rate:.1f}%")
    print(f"  Average rating: {sum(s.rating for s in standings) / len(standings):.0f}")

    print("\n✅ Tournament standings demonstration complete")


def demo_game_results():
    """Demonstrate game result tracking."""
    print("\n" + "=" * 60)
    print("Game Results Tracking Demo")
    print("=" * 60)

    # Create sample game results
    game_results = [
        GameResult(
            white_engine="Matrix0",
            black_engine="Stockfish",
            result="0-1",  # Black wins
            time_control="5000ms",
            opening="Sicilian Defense",
            moves=45,
            reason="Checkmate"
        ),
        GameResult(
            white_engine="Matrix0",
            black_engine="LC0",
            result="1/2-1/2",  # Draw
            time_control="5000ms",
            opening="Queen's Gambit",
            moves=67,
            reason="Stalemate"
        ),
        GameResult(
            white_engine="Stockfish",
            black_engine="Matrix0",
            result="1-0",  # White wins
            time_control="5000ms",
            opening="Ruy Lopez",
            moves=38,
            reason="Checkmate"
        )
    ]

    print("Recent Game Results:")
    print("White vs Black | Result | Moves | Opening")
    print("-" * 50)

    for result in game_results:
        opening_short = result.opening.replace(" Defense", "").replace("'", "")
        print("12"
              "7"
              "3d"
              "12")

    # Calculate statistics
    matrix0_games = len([r for r in game_results if "Matrix0" in [r.white_engine, r.black_engine]])
    matrix0_score = 0
    for result in game_results:
        if result.white_engine == "Matrix0":
            if result.result == "1-0":
                matrix0_score += 1
            elif result.result == "1/2-1/2":
                matrix0_score += 0.5
        elif result.black_engine == "Matrix0":
            if result.result == "0-1":
                matrix0_score += 1
            elif result.result == "1/2-1/2":
                matrix0_score += 0.5

    print(f"\nMatrix0 Performance:")
    print(f"  Games played: {matrix0_games}")
    print(f"  Score: {matrix0_score}/{matrix0_games} ({matrix0_score/matrix0_games*100:.1f}%)")

    print("\n✅ Game results demonstration complete")


def demo_integration_with_webui():
    """Demonstrate integration with the WebUI."""
    print("\n" + "=" * 60)
    print("WebUI Integration Demo")
    print("=" * 60)

    print("The tournament system integrates with the Matrix0 WebUI:")
    print()
    print("🎮 Tournament Management Features:")
    print("  • Create tournaments with multiple formats")
    print("  • Real-time tournament monitoring")
    print("  • Live game viewing during matches")
    print("  • Engine rating tracking (ELO + Glicko-2)")
    print("  • Historical tournament results")
    print("  • Performance analytics and statistics")
    print()
    print("📊 Rating System Integration:")
    print("  • Automatic rating updates after games")
    print("  • Glicko-2 rating system for uncertainty")
    print("  • Rating deviation tracking")
    print("  • Volatility measurements")
    print()
    print("🔧 Configuration Options:")
    print("  • Multiple tournament formats")
    print("  • Configurable time controls")
    print("  • Adjustable concurrency")
    print("  • Custom engine selection")
    print()
    print("The WebUI provides a complete tournament management interface")
    print("accessible through the 'Tournament' tab!")

    print("\n✅ WebUI integration demonstration complete")


def main():
    """Run all tournament system demonstrations."""
    print("Matrix0 Tournament System Demonstration")
    print("=" * 60)
    print("This demo showcases the advanced tournament management system")
    print("that enables competitive evaluation of chess engines.")
    print()

    # Run all demos
    demo_rating_system()
    demo_tournament_config()
    demo_tournament_simulation()
    demo_engine_standings()
    demo_game_results()
    demo_integration_with_webui()

    print("\n" + "=" * 60)
    print("🎉 All tournament system demonstrations completed!")
    print("=" * 60)
    print("\nThe tournament system provides:")
    print("• Advanced rating systems (ELO + Glicko-2)")
    print("• Multiple tournament formats (Round Robin, Swiss, Elimination)")
    print("• Comprehensive game result tracking")
    print("• Real-time tournament monitoring")
    print("• Engine performance analytics")
    print("• WebUI integration for easy management")
    print("\nTournament Features:")
    print("• Automated game execution with multiple engines")
    print("• Rating uncertainty tracking with Glicko-2")
    print("• Flexible time controls and concurrency settings")
    print("• Historical data and performance trends")
    print("• Integration with SSL-enhanced Matrix0")
    print("\nThe tournament system enables:")
    print("• Objective engine comparison and benchmarking")
    print("• Automated rating maintenance")
    print("• Performance regression detection")
    print("• Competitive analysis and improvement tracking")
    print("\nMatrix0's tournament system is now production-ready! 🏆")


if __name__ == "__main__":
    main()
