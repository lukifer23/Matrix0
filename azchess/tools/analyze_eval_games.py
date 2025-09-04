#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze PGN evaluation games and summarize draw causes, decisiveness, and patterns.
Usage:
  python -m azchess.tools.analyze_eval_games --dir data/eval_games --max-moves 240
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn


def _pos_key(board: chess.Board) -> Tuple[str, bool, bool, bool, bool, int | None]:
    """A normalized repetition key for the current position (similar to FEN without move clocks)."""
    return (
        board.board_fen(),
        board.turn,
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.ep_square,
    )


def analyze_pgn(path: Path, max_moves_cap: int | None = None) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        return {"file": str(path), "error": "empty or invalid PGN"}

    board = game.board()
    move_list: List[chess.Move] = []
    captures = 0
    checks = 0
    no_capture_streak = 0
    max_no_capture_streak = 0
    pos_seen = {}
    repetition_count = 0

    for node in game.mainline():
        move = node.move
        if move is None:
            break
        move_list.append(move)
        # repetition tracking before push
        key = _pos_key(board)
        pos_seen[key] = pos_seen.get(key, 0) + 1

        # apply move
        was_capture = board.is_capture(move)
        board.push(move)
        if was_capture:
            captures += 1
            no_capture_streak = 0
        else:
            no_capture_streak += 1
            if no_capture_streak > max_no_capture_streak:
                max_no_capture_streak = no_capture_streak
        if board.is_check():
            checks += 1
        # repetition tracking after push
        key2 = _pos_key(board)
        c2 = pos_seen.get(key2, 0)
        if c2 >= 2:  # this state seen before
            repetition_count += 1

    result_header = game.headers.get("Result", "*")
    termination = game.headers.get("Termination", "")
    moves_played = len(move_list)

    # Simple end material tally
    piece_map = board.piece_map()
    white_pcs = sum(1 for _, p in piece_map.items() if p.color == chess.WHITE)
    black_pcs = sum(1 for _, p in piece_map.items() if p.color == chess.BLACK)

    inferred_cap_draw = False
    if max_moves_cap is not None and moves_played >= max_moves_cap and result_header == "1/2-1/2":
        inferred_cap_draw = True

    return {
        "file": str(path),
        "result": result_header,
        "termination": termination or ("cap_draw" if inferred_cap_draw else ""),
        "moves": moves_played,
        "captures": captures,
        "checks": checks,
        "max_no_capture_streak": max_no_capture_streak,
        "repetition_hits": repetition_count,
        "end_white_pieces": white_pcs,
        "end_black_pieces": black_pcs,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze enhanced eval PGNs")
    ap.add_argument("--dir", type=str, default="data/eval_games", help="Directory of PGN files")
    ap.add_argument("--max-moves", type=int, default=240, help="Move cap to infer cap draws")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("*.pgn"))
    if not files:
        print(f"No PGNs found in {d}")
        return

    rows: List[Dict[str, object]] = []
    for p in files:
        try:
            rows.append(analyze_pgn(p, max_moves_cap=args.max_moves))
        except Exception as e:
            rows.append({"file": str(p), "error": str(e)})

    # Aggregate
    draws = sum(1 for r in rows if r.get("result") == "1/2-1/2")
    wins_w = sum(1 for r in rows if r.get("result") == "1-0")
    wins_b = sum(1 for r in rows if r.get("result") == "0-1")
    avg_moves = sum(r.get("moves", 0) for r in rows) / max(1, len(rows))
    avg_caps = sum(r.get("captures", 0) for r in rows) / max(1, len(rows))
    avg_checks = sum(r.get("checks", 0) for r in rows) / max(1, len(rows))
    avg_streak = sum(r.get("max_no_capture_streak", 0) for r in rows) / max(1, len(rows))
    cap_draws = sum(1 for r in rows if r.get("termination") == "cap_draw")

    print("Evaluation Games Summary")
    print(f"  Files: {len(rows)}  W: {wins_w}  B: {wins_b}  D: {draws}  CapDraws~: {cap_draws}")
    print(f"  Avg Moves: {avg_moves:.1f}  Avg Captures: {avg_caps:.1f}  Avg Checks: {avg_checks:.1f}  Avg Max No-Capture Streak: {avg_streak:.1f}")

    # Print top 5 with longest no-capture streaks (potential shuffling)
    long_streaks = sorted(rows, key=lambda r: r.get("max_no_capture_streak", 0), reverse=True)[:5]
    print("\nTop shuffle-prone games (long no-capture streak):")
    for r in long_streaks:
        print(f"  {Path(r['file']).name}: streak={r.get('max_no_capture_streak')} moves={r.get('moves')} term={r.get('termination','')} result={r.get('result')}")

    # Print any errors
    errs = [r for r in rows if r.get("error")]
    if errs:
        print("\nErrors:")
        for r in errs[:10]:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()

