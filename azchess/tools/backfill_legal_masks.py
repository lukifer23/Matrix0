from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import chess

from azchess.encoding import move_encoder


def compute_legal_masks(states: np.ndarray, start: int = 0, end: int | None = None) -> np.ndarray:
    """Compute legal masks for a slice of encoded states (N, P, 8, 8).

    Reconstructs minimal board info from planes:
    - Piece planes (0..11) and side-to-move plane (12)
    - Castling rights (13..16)
    """
    if end is None:
        end = states.shape[0]
    N = int(end - start)
    masks = np.zeros((N, 4672), dtype=np.uint8)
    for idx, i in enumerate(range(start, end)):
        planes = states[i]
        board = chess.Board.empty()
        # side to move
        try:
            board.turn = bool(planes[12, 0, 0] > 0.5)
        except Exception:
            board.turn = True
        # castling rights
        try:
            if planes[13, 0, 0] > 0.5:
                board.castling_rights |= chess.CASTLING_WHITE_K
            if planes[14, 0, 0] > 0.5:
                board.castling_rights |= chess.CASTLING_WHITE_Q
            if planes[15, 0, 0] > 0.5:
                board.castling_rights |= chess.CASTLING_BLACK_K
            if planes[16, 0, 0] > 0.5:
                board.castling_rights |= chess.CASTLING_BLACK_Q
        except Exception:
            pass
        # pieces
        piece_map = {0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING}
        try:
            for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
                base = color_idx * 6
                for k in range(6):
                    plane = planes[base + k]
                    piece_type = piece_map[k]
                    # fast iteration over nonzero cells
                    rr, cc = np.where(plane > 0.5)
                    for r, f in zip(rr.tolist(), cc.tolist()):
                        sq = chess.square(f, 7 - r)
                        board.set_piece_at(sq, chess.Piece(piece_type, color))
        except Exception:
            pass
        # generate mask
        try:
            m = move_encoder.get_legal_actions(board)
            masks[idx] = m.astype(np.uint8, copy=False)
        except Exception:
            pass
    return masks


def process_file(path: Path, overwrite: bool, chunk_size: int = 2048) -> bool:
    try:
        # Avoid mmap for object arrays in external NPZs
        with np.load(path, allow_pickle=True) as data:
            if 'legal_mask' in data and not overwrite:
                return False
            # Support external NPZs that use 'positions' key
            if 's' in data:
                s = data['s']
            elif 'positions' in data:
                s = data['positions']
            else:
                print(f"[skip] {path.name}: missing 's'/'positions'")
                return False
            N = int(s.shape[0])
            # Create temp on-disk memmap for large masks
            tmp_mask_path = path.with_suffix('.legal_mask.tmp.npy')
            lm_mem = np.memmap(tmp_mask_path, dtype=np.uint8, mode='w+', shape=(N, 4672))
            # chunked compute
            for start in range(0, N, chunk_size):
                end = min(N, start + chunk_size)
                lm_chunk = compute_legal_masks(s, start, end)
                lm_mem[start:end] = lm_chunk
            del lm_mem  # flush to disk

            payload = {k: data[k] for k in data.files}
            # ensure plain ndarray for mask when saving
            payload['legal_mask'] = np.load(tmp_mask_path, mmap_mode='r').astype(np.uint8, copy=False)
        # Save atomically by writing to a temp and replacing
        tmp = path.parent / (path.stem + '.tmp.npz')
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        try:
            Path(tmp_mask_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"[error] {path.name}: save failed: {e}")
            return False
        return True
    except Exception as e:
        print(f"[error] {path.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Backfill legal_mask into NPZ shards")
    ap.add_argument('--dir', type=str, default='data/selfplay', help='Directory of NPZ shards to backfill')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing legal_mask if present')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files (0 = all)')
    ap.add_argument('--chunk-size', type=int, default=2048, help='Processing chunk size for large files')
    args = ap.parse_args()

    d = Path(args.dir)
    paths = sorted([p for p in d.glob('*.npz') if not p.name.endswith('.tmp.npz')])
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    updated = 0
    for p in paths:
        print(f"Backfilling legal_mask -> {p.name}")
        if process_file(p, args.overwrite, args.chunk_size):
            updated += 1
    print(f"Updated {updated}/{len(paths)} files in {d}")


if __name__ == '__main__':
    main()


