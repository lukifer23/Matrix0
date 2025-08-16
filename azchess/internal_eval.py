
from __future__ import annotations

import chess
import torch

from .config import Config, select_device
from .model import PolicyValueNet
from .mcts import MCTS, MCTSConfig

def play_match(ckpt_a: str, ckpt_b: str, games: int, cfg: Config) -> float:
    """
    Plays a match between two model checkpoints.
    Returns the score of model A.
    """
    device = select_device(cfg.get("device", "auto"))
    mcfg = MCTSConfig(**cfg.eval())
    
    model_a = PolicyValueNet.from_config(cfg.model()).to(device)
    model_b = PolicyValueNet.from_config(cfg.model()).to(device)
    
    state_a = torch.load(ckpt_a, map_location=device)
    state_b = torch.load(ckpt_b, map_location=device)
    
    model_a.load_state_dict(state_a.get("model_ema", state_a["model"]))
    model_b.load_state_dict(state_b.get("model_ema", state_b["model"]))
    
    mcts_a = MCTS(model_a, mcfg, device)
    mcts_b = MCTS(model_b, mcfg, device)

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
