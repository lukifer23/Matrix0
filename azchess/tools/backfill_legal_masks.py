from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import chess

from azchess.encoding import move_encoder


def compute_legal_masks(states: np.ndarray) -> np.ndarray:
    """Compute legal masks for encoded states array (N, P, 8, 8).

    We reconstruct a board position minimally from planes sufficient for legality:
    - Piece planes (0..11) for piece placement and side to move (12)
    - Castling rights (13..16). En-passant not derived; rare moves may be missing.
    """
    N = states.shape[0]
    masks = np.zeros((N, 4672), dtype=np.uint8)
    for i in range(N):
        planes = states[i]
        board = chess.Board.empty()
        # side to move
        board.turn = bool(planes[12, 0, 0] > 0.5)
        # castling rights
        if planes[13, 0, 0] > 0.5:
            board.castling_rights |= chess.CASTLING_WHITE_K
        if planes[14, 0, 0] > 0.5:
            board.castling_rights |= chess.CASTLING_WHITE_Q
        if planes[15, 0, 0] > 0.5:
            board.castling_rights |= chess.CASTLING_BLACK_K
        if planes[16, 0, 0] > 0.5:
            board.castling_rights |= chess.CASTLING_BLACK_Q
        # pieces
        piece_map = {
            0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING
        }
        for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
            base = color_idx * 6
            for k in range(6):
                plane = planes[base + k]
                piece_type = piece_map[k]
                for r in range(8):
                    for f in range(8):
                        if plane[r, f] > 0.5:
                            sq = chess.square(f, 7 - r)
                            board.set_piece_at(sq, chess.Piece(piece_type, color))
        # generate mask
        try:
            m = move_encoder.get_legal_actions(board)
            masks[i] = m.astype(np.uint8, copy=False)
        except Exception:
            # leave zeros on failure
            pass
    return masks


def process_file(path: Path, overwrite: bool) -> bool:
    try:
        with np.load(path) as data:
            if 'legal_mask' in data:
                if not overwrite:
                    return False
            s = data['s']
            lm = compute_legal_masks(s)
            payload = {k: data[k] for k in data.files}
            payload['legal_mask'] = lm
        # Save atomically by writing to a temp and replacing
        tmp = path.with_suffix('.npz.tmp')
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Backfill legal_mask into NPZ shards")
    ap.add_argument('--dir', type=str, default='data/selfplay', help='Directory of NPZ shards to backfill')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing legal_mask if present')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files (0 = all)')
    args = ap.parse_args()

    d = Path(args.dir)
    paths = sorted([p for p in d.glob('*.npz')])
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    updated = 0
    for p in paths:
        if process_file(p, args.overwrite):
            updated += 1
    print(f"Updated {updated}/{len(paths)} files in {d}")


if __name__ == '__main__':
    main()


