from __future__ import annotations

import argparse
from typing import List, Tuple

import chess
import numpy as np

from .encoding import move_to_index, MoveEncoder


def random_board(plies: int = 30, seed: int = 42) -> chess.Board:
    rng = np.random.default_rng(seed)
    b = chess.Board()
    for _ in range(plies):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        mv = moves[int(rng.integers(0, len(moves)))]
        b.push(mv)
    return b


def validate_board(b: chess.Board) -> Tuple[int, int, List[str]]:
    legal = list(b.legal_moves)
    idxs = [move_to_index(b, m) for m in legal]
    unique = len(set(idxs))
    collisions = len(idxs) - unique
    errors: List[str] = []
    # Sanity range
    for i in idxs:
        if not (0 <= i < 64 * 73):
            errors.append("index_out_of_range")
            break
    # Collision check
    if collisions > 0:
        errors.append(f"collisions={collisions}")
    return len(legal), collisions, errors


def validate_edge_cases() -> List[str]:
    errs: List[str] = []
    enc = MoveEncoder()
    # Castling across check scenarios
    positions = [
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8c8"),
    ]
    for fen, uci in positions:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if mv in b.legal_moves:
            idx = enc.encode_move(b, mv)
            dm = enc.decode_move(b, idx)
            if dm not in b.legal_moves:
                errs.append(f"castling_illegal_after_decode:{uci}")
    # En passant
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    b.push(chess.Move.from_uci("d7d5"))
    mv = chess.Move.from_uci("e4d5")
    if mv in b.legal_moves:
        idx = enc.encode_move(b, mv)
        dm = enc.decode_move(b, idx)
        if dm not in b.legal_moves:
            errs.append("en_passant_decode_illegal")
    # Promotions (queen via rays, underpromotions explicit)
    for promo in ["q", "n", "b", "r"]:
        test = chess.Board("8/3P4/8/8/8/8/8/8 w - - 0 1")
        uci = f"d7d8{promo}"
        mv = chess.Move.from_uci(uci)
        if mv in test.legal_moves or promo == "q":
            idx = enc.encode_move(test, mv)
            dm = enc.decode_move(test, idx)
            if dm.to_square != mv.to_square or dm.from_square != mv.from_square:
                errs.append(f"promo_mismatch:{uci}")
    return errs


def main():
    ap = argparse.ArgumentParser(description="Validate 4672 move mapping")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--plies", type=int, default=30)
    ap.add_argument("--report-collisions", action="store_true")
    args = ap.parse_args()

    total_legal = 0
    total_collisions = 0
    issues = 0
    for s in range(args.samples):
        b = random_board(plies=args.plies, seed=42 + s)
        legal, collisions, errs = validate_board(b)
        total_legal += legal
        total_collisions += collisions
        if errs:
            issues += 1
    edge_errs = validate_edge_cases()
    print(f"Boards tested={args.samples}, total_legal={total_legal}, total_collisions={total_collisions}, issue_boards={issues}, edge_issues={len(edge_errs)}")
    if edge_errs:
        for e in edge_errs[:10]:
            print("edge_issue:", e)


if __name__ == "__main__":
    main()
