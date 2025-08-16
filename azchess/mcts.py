from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import chess
import torch

from .encoding import encode_board, move_to_index


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    tt_capacity: int = 200000
    selection_jitter: float = 0.0


class Node:
    __slots__ = ("parent", "prior", "n", "w", "q", "children", "move")

    def __init__(self, prior: float = 0.0, move: Optional[chess.Move] = None, parent: Optional["Node"] = None):
        self.parent: Optional[Node] = parent
        self.prior: float = prior
        self.n: int = 0
        self.w: float = 0.0
        self.q: float = 0.0
        self.children: Dict[chess.Move, Node] = {}
        self.move = move

    def expand(self, board: chess.Board, policy_logits: np.ndarray) -> None:
        legal = list(board.legal_moves)
        if not legal:
            return
        # Softmax over legal moves using action indices
        logits = []
        moves = []
        for m in legal:
            idx = move_to_index(board, m)
            logits.append(policy_logits[idx])
            moves.append(m)
        logits = np.array(logits, dtype=np.float32)
        # Stable softmax
        logits -= np.max(logits)
        exps = np.exp(logits)
        z = np.sum(exps)
        if z <= 0 or not np.isfinite(z):
            probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float32)
        else:
            probs = exps / z
        for m, p in zip(moves, probs):
            self.children[m] = Node(prior=float(p), move=m, parent=self)

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, cfg: MCTSConfig, device: str = "cpu"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.nn_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.tt: "OrderedDict[str, Node]" = OrderedDict()
        self.tt_capacity = getattr(self.cfg, "tt_capacity", 200000)

    @torch.no_grad()
    def run(self, board: chess.Board, num_simulations: Optional[int] = None) -> Tuple[Dict[chess.Move, int], np.ndarray, float]:
        root = Node()
        p_logits, v = self._infer(board)
        root.expand(board, p_logits)
        # Register root children in transposition table
        for m, child in list(root.children.items()):
            b2 = board.copy(); b2.push(m)
            fen2 = b2.fen()
            if fen2 in self.tt:
                node_ref = self.tt.pop(fen2)
                self.tt[fen2] = node_ref
                root.children[m] = node_ref
            else:
                self._tt_put(fen2, child)
        self._add_dirichlet(root)

        sims = num_simulations or self.cfg.num_simulations
        for _ in range(sims):
            node, path, states = self._select(board.copy(), root)
            if not node.is_expanded():
                p_logits, v_leaf = self._infer(states[-1])
                node.expand(states[-1], p_logits)
                # Register new children in TT
                cur_board = states[-1]
                for m, child in list(node.children.items()):
                    b2 = cur_board.copy(); b2.push(m)
                    fen2 = b2.fen()
                    if fen2 in self.tt:
                        node_ref = self.tt.pop(fen2)
                        self.tt[fen2] = node_ref
                        node.children[m] = node_ref
                    else:
                        self._tt_put(fen2, child)
            else:
                # If expanded but reached terminal state, use terminal value
                v_leaf = self._terminal_value(states[-1])
            self._backpropagate(path, v_leaf)

        visit_counts: Dict[chess.Move, int] = {m: c.n for m, c in root.children.items()}
        policy = self._policy_from_root(root, board)
        return visit_counts, policy, v

    def _policy_from_root(self, root: Node, board: chess.Board) -> np.ndarray:
        pi = np.zeros(4672, dtype=np.float32)
        total = sum(c.n for c in root.children.values())
        total = max(total, 1)
        for m, child in root.children.items():
            idx = move_to_index(board, m)
            pi[idx] = child.n / total
        return pi

    def _select(self, board: chess.Board, root: Node):
        node = root
        path = [root]
        states = [board]
        while node.is_expanded():
            best_score = -1e9
            best_move = None
            best_child = None
            sum_n = math.sqrt(sum(c.n for c in node.children.values()))
            jitter = self.cfg.selection_jitter
            for m, c in node.children.items():
                u = self.cfg.cpuct * c.prior * (sum_n / (1 + c.n))
                score = c.q + u
                if jitter > 0.0:
                    score += random.random() * jitter
                if score > best_score:
                    best_score = score
                    best_move, best_child = m, c
            if best_child is None:
                break
            board.push(best_move)
            # Link to TT node if any, else register
            fen2 = board.fen()
            if fen2 in self.tt and self.tt[fen2] is not best_child:
                node_ref = self.tt.pop(fen2)
                self.tt[fen2] = node_ref
                node.children[best_move] = node_ref
                best_child = node.children[best_move]
            else:
                self._tt_put(fen2, best_child)
            node = best_child
            path.append(node)
            states.append(board.copy())
            if board.is_game_over():
                break
        return node, path, states

    def _backpropagate(self, path: List[Node], value: float) -> None:
        # Value is from the perspective of the current player at the leaf.
        v = value
        for node in reversed(path):
            node.n += 1
            node.w += v
            node.q = node.w / node.n
            v = -v

    def _add_dirichlet(self, root: Node) -> None:
        if not root.children:
            return
        alpha = self.cfg.dirichlet_alpha
        frac = self.cfg.dirichlet_frac
        if frac <= 0.0:
            return
        noise = np.random.dirichlet([alpha] * len(root.children))
        for (m, child), n in zip(root.children.items(), noise):
            child.prior = child.prior * (1 - frac) + float(n) * frac

    @torch.no_grad()
    def _infer(self, board: chess.Board):
        import numpy as np
        from .encoding import encode_board

        arr = encode_board(board)
        fen = board.fen()
        if fen in self.nn_cache:
            return self.nn_cache[fen]
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.autocast(self.device if self.device != "cpu" else "cpu", enabled=(self.device != "cpu")):
            p, v = self.model(x)
        out = (p[0].detach().cpu().numpy(), float(v.item()))
        self.nn_cache[fen] = out
        return out

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _tt_put(self, key: str, node: Node) -> None:
        self.tt[key] = node
        if len(self.tt) > self.tt_capacity:
            self.tt.popitem(last=False)
