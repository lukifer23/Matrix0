
from __future__ import annotations

import chess
from .config import Config
from .utils.model_loader import load_model_and_mcts

def play_match(ckpt_a: str, ckpt_b: str, games: int, cfg: Config) -> float:
    """
    Plays a match between two model checkpoints.
    Returns the score of model A.
    """
    model_a, mcts_a = load_model_and_mcts(cfg, ckpt_a)
    model_b, mcts_b = load_model_and_mcts(cfg, ckpt_b)

    score = 0.0
    for g in range(games):
        board = chess.Board()
        # Alternate colors
        engines = [mcts_a, mcts_b] if g % 2 == 0 else [mcts_b, mcts_a]
        
        while not board.is_game_over(claim_draw=True):
            mcts = engines[0] if board.turn == chess.WHITE else engines[1]
            visits, _, _ = mcts.run(board)
            move = max(visits.items(), key=lambda kv: kv[1])[0]
            board.push(move)
            
        result = board.result(claim_draw=True)
        if result == "1-0":
            score += 1.0 if g % 2 == 0 else 0.0
        elif result == "0-1":
            score += 0.0 if g % 2 == 0 else 1.0
        else:
            score += 0.5
            
    return score
