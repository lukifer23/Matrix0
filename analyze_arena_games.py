#!/usr/bin/env python3
"""
Analyze arena games to understand MCTS behavior and move quality.
This script will help debug why MCTS is returning no visits in certain positions.
"""

import argparse
import os
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azchess.arena import _arena_run_one_game, _arena_worker_init
from azchess.config import Config
from azchess.mcts import MCTS, MCTSConfig


def analyze_single_game(ckpt_a_path: str, ckpt_b_path: str, game_idx: int = 0, 
                       num_sims: int = 400, batch_size: int = 2, device: str = "auto"):
    """Analyze a single game with detailed move logging."""
    
    # Load configuration
    cfg = Config.load("config.yaml")
    cfg_dict = cfg.to_dict()
    
    # Create MCTS instances directly
    from azchess.config import select_device
    from azchess.model import PolicyValueNet
    
    device = select_device(device)
    mcfg_dict = dict(cfg.mcts())
    mcfg_dict.update(
        {
            "num_simulations": num_sims,
            "selection_jitter": float(cfg.selfplay().get("selection_jitter", 0.0)),
            "batch_size": batch_size,
            "tt_cleanup_frequency": int(cfg.mcts().get("tt_cleanup_frequency", 500)),
        }
    )
    mcfg = MCTSConfig.from_dict(mcfg_dict)
    
    # Load models
    model_a = PolicyValueNet.from_config(cfg.model()).to(device)
    model_b = PolicyValueNet.from_config(cfg.model()).to(device)
    
    import torch
    sa = torch.load(ckpt_a_path, map_location=device, weights_only=False)
    sb = torch.load(ckpt_b_path, map_location=device, weights_only=False)
    model_a.load_state_dict(sa.get("model_ema", sa.get("model", sa)))
    model_b.load_state_dict(sb.get("model_ema", sb.get("model", sb)))
    
    # Create MCTS instances
    mcts_a = MCTS(model_a, mcfg, device)
    mcts_b = MCTS(model_b, mcfg, device)
    
    # Create a custom game runner that captures all moves
    board = chess.Board()
    moves_count = 0
    a_is_white = (game_idx % 2 == 0)
    engines = ("A (trained)", "B (untrained)") if a_is_white else ("B (untrained)", "A (trained)")
    
    print(f"🎮 Analyzing Game {game_idx + 1}")
    print(f"📊 White: {engines[0]}, Black: {engines[1]}")
    print(f"⚙️  MCTS Simulations: {num_sims}")
    print("=" * 60)
    
    game_moves = []
    max_moves = 300
    
    while (not board.is_game_over(claim_draw=True)) and (moves_count < max_moves):
        stm_white = board.turn == chess.WHITE
        engine_name = engines[0] if stm_white else engines[1]
        mcts = mcts_a if (a_is_white == stm_white) else mcts_b
        
        # Get current position info
        legal_moves = list(board.legal_moves)
        print(f"\nMove {moves_count + 1}: {len(legal_moves)} legal moves")
        print(f"Position: {board.fen()}")
        print(f"Engine: {engine_name}")
        
        # Run MCTS
        try:
            visits, pi, vroot = mcts.run(board)
            
            if visits:
                # MCTS found moves
                move = max(visits.items(), key=lambda kv: kv[1])[0]
                visit_counts = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)
                top_moves = visit_counts[:3]
                
                print(f"✅ MCTS visits: {dict(top_moves)}")
                print(f"🎯 Selected: {move} (SAN: {board.san(move)})")
                print(f"📊 Root value: {vroot:.3f}")
                
            else:
                # MCTS failed - use policy-based fallback for analysis fidelity
                try:
                    from azchess.encoding import move_to_index
                    best = None
                    best_score = -1.0
                    for mv in legal_moves:
                        try:
                            idx = move_to_index(board, mv)
                            score = float(pi[idx]) if 0 <= idx < len(pi) else 0.0
                        except Exception:
                            score = 0.0
                        if score > best_score:
                            best_score = score
                            best = mv
                    move = best or np.random.choice(legal_moves)
                except Exception:
                    move = np.random.choice(legal_moves)
                print(f"❌ MCTS returned no visits!")
                print(f"⚠️  Using policy-based fallback: {move} (SAN: {board.san(move)})")
                print(f"🔍 This position needs investigation!")
                
                # Let's analyze why MCTS failed
                print(f"📋 Legal moves: {[board.san(m) for m in legal_moves[:10]]}")
                if len(legal_moves) > 10:
                    print(f"   ... and {len(legal_moves) - 10} more")
                
        except Exception as e:
            print(f"💥 MCTS error: {e}")
            try:
                from azchess.encoding import move_to_index
                best = None
                best_score = -1.0
                for mv in legal_moves:
                    try:
                        idx = move_to_index(board, mv)
                        score = float(pi[idx]) if 0 <= idx < len(pi) else 0.0
                    except Exception:
                        score = 0.0
                    if score > best_score:
                        best_score = score
                        best = mv
                move = best or np.random.choice(legal_moves)
            except Exception:
                move = np.random.choice(legal_moves)
            print(f"⚠️  Using policy-based fallback: {move}")
        
        # Record the move
        game_moves.append({
            'move_number': moves_count + 1,
            'move': move,
            'san': board.san(move),
            'fen': board.fen(),
            'engine': engine_name,
            'mcts_success': bool(visits) if 'visits' in locals() else False
        })
        
        # Make the move
        board.push(move)
        moves_count += 1
        
        # Check for game end
        if board.is_checkmate():
            print(f"\n🏆 Checkmate! {engine_name} wins!")
            break
        elif board.is_stalemate():
            print(f"\n🤝 Stalemate!")
            break
        elif board.is_insufficient_material():
            print(f"\n🤝 Insufficient material!")
            break
        elif board.is_fifty_moves():
            print(f"\n🤝 Fifty-move rule!")
            break
        elif board.is_repetition():
            print(f"\n🤝 Repetition!")
            break
    
    # Game summary
    print("\n" + "=" * 60)
    print("📋 GAME SUMMARY")
    print("=" * 60)
    
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        print(f"🏁 Result: {result}")
    else:
        print(f"🏁 Game ended after {moves_count} moves (max reached)")
    
    print(f"📊 Total moves: {moves_count}")
    
    # Analyze MCTS failures
    mcts_failures = [m for m in game_moves if not m['mcts_success']]
    if mcts_failures:
        print(f"\n❌ MCTS FAILURES: {len(mcts_failures)}")
        print("=" * 60)
        for failure in mcts_failures:
            print(f"Move {failure['move_number']}: {failure['san']}")
            print(f"Engine: {failure['engine']}")
            print(f"FEN: {failure['fen']}")
            print("-" * 40)
    else:
        print("\n✅ No MCTS failures detected")
    
    return game_moves, board


def main():
    parser = argparse.ArgumentParser(description="Analyze arena games for MCTS debugging")
    parser.add_argument("--ckpt_a", required=True, help="Path to model A checkpoint")
    parser.add_argument("--ckpt_b", required=True, help="Path to model B checkpoint")
    parser.add_argument("--games", type=int, default=1, help="Number of games to analyze")
    parser.add_argument("--num-sims", type=int, default=400, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    print("🔍 ARENA GAME ANALYZER")
    print("=" * 60)
    print(f"📁 Model A: {args.ckpt_a}")
    print(f"📁 Model B: {args.ckpt_b}")
    print(f"🎮 Games to analyze: {args.games}")
    print(f"⚙️  MCTS simulations: {args.num_sims}")
    print("=" * 60)
    
    all_games = []
    
    for game_idx in range(args.games):
        try:
            moves, final_board = analyze_single_game(
                args.ckpt_a, args.ckpt_b, game_idx, 
                args.num_sims, args.batch_size, args.device
            )
            all_games.append(moves)
            
            if game_idx < args.games - 1:
                print("\n" + "🔄" * 20 + " NEXT GAME " + "🔄" * 20 + "\n")
                
        except Exception as e:
            print(f"💥 Error analyzing game {game_idx + 1}: {e}")
            continue
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("📊 OVERALL ANALYSIS")
    print("=" * 60)
    
    total_moves = sum(len(game) for game in all_games)
    total_mcts_failures = sum(sum(1 for move in game if not move['mcts_success']) for game in all_games)
    
    print(f"🎮 Total games analyzed: {len(all_games)}")
    print(f"📊 Total moves: {total_moves}")
    print(f"❌ Total MCTS failures: {total_mcts_failures}")
    
    if total_moves > 0:
        failure_rate = (total_mcts_failures / total_moves) * 100
        print(f"📈 MCTS failure rate: {failure_rate:.1f}%")
        
        if failure_rate > 10:
            print("\n🚨 HIGH MCTS FAILURE RATE DETECTED!")
            print("This suggests issues with:")
            print("  - Model inference (NaN/Inf values)")
            print("  - MCTS tree expansion")
            print("  - Memory pressure")
            print("  - Configuration parameters")
        elif failure_rate > 5:
            print("\n⚠️  MODERATE MCTS FAILURE RATE")
            print("Some investigation recommended")
        else:
            print("\n✅ Low MCTS failure rate - system working well")


if __name__ == "__main__":
    main()
