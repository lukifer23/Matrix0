from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import chess
import torch
import logging

try:  # Optional dependency for memory monitoring
    import psutil  # Added for memory monitoring
    psutil_available = True
except ImportError:  # pragma: no cover - exercised in tests via patching
    psutil_available = False

logger = logging.getLogger(__name__)

from .encoding import encode_board, move_to_index, MoveEncoder


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Tuple[np.ndarray, float]]:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Tuple[np.ndarray, float]):
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

@dataclass
class MCTSConfig:
    num_simulations: int = 800
    cpuct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    dirichlet_plies: int = 16
    tt_capacity: int = 2_000_000
    selection_jitter: float = 0.01
    tt_cleanup_frequency: int = 5000
    tt_memory_limit_mb: int = 2048
    batch_size: int = 32
    fpu: float = 0.5  # First-Play Urgency
    parent_q_init: bool = True # Initialize child Q with parent Q
    draw_penalty: float = -0.1  # Slight draw penalty to reduce draws
    virtual_loss: float = 1.0  # Penalty applied to in-flight edges during batched selection
    # Optional cpuct schedule by ply (linear from start to end over cpuct_plies)
    cpuct_start: Optional[float] = None
    cpuct_end: Optional[float] = None
    cpuct_plies: int = 0
    # If True, NN value is from absolute White's perspective and must be flipped
    # to side-to-move perspective for MCTS/backprop/resign logic
    value_from_white: bool = False
    # Child pruning
    max_children: int = 0            # keep top-K children by prior; 0 disables
    min_child_prior: float = 0.0     # drop children with prior < threshold after normalization
    # Optional optimizations
    legal_softmax: bool = False      # softmax over legal moves only
    encoder_cache: bool = True       # use MoveEncoder caching
    tt_cleanup_interval_s: int = 5   # periodic TT cleanup interval (seconds)
    no_instant_backtrack: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSConfig":
        """Create config from dict, warning on unknown keys."""
        known = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - known
        if unknown:
            logger.warning(f"Unknown MCTSConfig keys: {sorted(unknown)}")
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class Node:
    __slots__ = ("parent", "prior", "n", "w", "q", "children", "move", "expanded")

    def __init__(self, prior: float = 0.0, move: Optional[chess.Move] = None, parent: Optional["Node"] = None):
        self.parent: Optional[Node] = parent
        self.prior: float = prior
        self.n: int = 0
        self.w: float = 0.0
        self.q: float = 0.0
        self.children: Dict[chess.Move, Node] = {}
        self.move = move
        self.expanded = False

    def expand(self, board: chess.Board, p_logits: np.ndarray, encoder: Optional[MoveEncoder] = None, legal_only: bool = False) -> None:
        """Expand this node with children for all legal moves."""
        if self.is_expanded():
            return
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return
        
        logits = p_logits.astype(np.float32, copy=False)
        
        # Handle edge cases: NaN or infinite values
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            # Fallback to uniform distribution
            lp = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        else:
            # Convert logits to probabilities
            if legal_only:
                # Softmax over legal indices only
                if encoder is not None:
                    idxs = [encoder.encode_move(board, m) for m in legal_moves]
                else:
                    idxs = [move_to_index(board, m) for m in legal_moves]
                sel = torch.from_numpy(logits[idxs])
                sel = torch.softmax(sel, dim=-1).numpy()
                probs = None
            else:
                probs = torch.from_numpy(logits)
                probs = torch.softmax(probs, dim=-1).numpy()

            # Check if policy is too uniform (degenerate case)
            policy_entropy = -np.sum(probs * np.log(probs + 1e-8))
            max_entropy = np.log(len(legal_moves))
            entropy_ratio = policy_entropy / max_entropy
            
            # If policy is too uniform, add noise to encourage exploration (quiet log)
            if entropy_ratio > 0.9:
                logger.debug(f"Policy too uniform (entropy ratio: {entropy_ratio:.3f}), adding exploration noise")
                # Add small random noise to break uniformity
                noise = np.random.normal(0, 0.1, probs.shape)
                probs = probs + noise
                probs = np.maximum(probs, 1e-8)  # Ensure positive
                probs = probs / probs.sum()  # Renormalize

            legal_priors: List[float] = []
            for i, move in enumerate(legal_moves):
                try:
                    if legal_only:
                        prior = float(sel[i])
                    else:
                        idx = encoder.encode_move(board, move) if encoder is not None else move_to_index(board, move)
                        prior = float(probs[idx])
                    # Ensure prior is valid
                    if np.isnan(prior) or np.isinf(prior) or prior < 0:
                        prior = 0.0
                    legal_priors.append(prior)
                except Exception:
                    legal_priors.append(0.0)

            lp = np.asarray(legal_priors, dtype=np.float32)
            total_lp = lp.sum()
            
            # Ensure we have valid probabilities
            if total_lp > 0 and not np.isnan(total_lp) and not np.isinf(total_lp):
                lp = lp / total_lp
            else:
                lp = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)

        for move, prior in zip(legal_moves, lp):
            child = Node(prior=float(prior), move=move, parent=self)
            if self.parent and self.parent.q != 0.0:
                child.q = -self.parent.q
            self.children[move] = child
        
        self.expanded = True
    
    def is_expanded(self) -> bool:
        return self.expanded

    def get_ucb_score(self, parent_visits: int, cpuct: float) -> float:
        """Calculate UCB score for node selection."""
        if self.n == 0:
            # Optimistic value for unvisited nodes - encourage exploration
            return cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.n)
        return self.q + cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.n)


class MCTS:
    def __init__(self, model, cfg: MCTSConfig, device: str = "cpu", inference_backend: any | None = None):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.inference_backend = inference_backend
        self.nn_cache = LRUCache(capacity=50000)
        self.tt: "OrderedDict[str, Node]" = OrderedDict()
        
        self.simulations_run = 0
        self.tt_hits = 0
        self.tt_misses = 0
        self._last_cleanup = 0
        self._last_sims_run = 0
        self._memory_cleanup_threshold = 80  # Memory percentage threshold
        # For debug/inspection in arena and tools
        self._last_root: Optional[Node] = None
        # Guard to defer cleanup during bulk inserts into TT
        self._in_bulk_put: bool = False
        # Move encoder cache
        self._enc: Optional[MoveEncoder] = MoveEncoder() if bool(cfg.encoder_cache) else None
        self._last_cleanup_wall: float = time.time()

    @torch.no_grad()
    def run(self, board: chess.Board, num_simulations: Optional[int] = None, ply: Optional[int] = None) -> Tuple[Dict[chess.Move, int], np.ndarray, float]:
        if board.is_game_over():
            return {}, np.zeros(4672, dtype=np.float32), self._terminal_value(board)

        key = board._transposition_key()
        root = self._tt_get(key)
        if root is None:
            root = Node()
            p_logits, v = self._infer(board)
            root.expand(board, p_logits, encoder=self._enc, legal_only=self.cfg.legal_softmax)
            self._prune_children(root)
            self._tt_put(key, root)
        else:
            cached = self.nn_cache.get(key)
            if cached is None:
                # Cache miss: recompute value without expanding existing root
                p_logits, v = self._infer(board)
                # Ensure recomputed result is stored for future lookups
                self.nn_cache.put(key, (p_logits, v))
            else:
                # Use cached value when available
                _, v = cached

        # Optionally gate Dirichlet noise by ply (apply in early game only)
        dirichlet_plies = getattr(self.cfg, 'dirichlet_plies', None)
        if dirichlet_plies is None or (ply is None) or (ply < int(dirichlet_plies)):
            self._add_dirichlet(root)

        sims_to_run = num_simulations if num_simulations is not None else self.cfg.num_simulations
        
        sims_run_this_turn = 0
        while sims_run_this_turn < sims_to_run:
            batch_size = min(self.cfg.batch_size, sims_to_run - sims_run_this_turn)
            leaves: List[Tuple[Node, List[Node], chess.Board]] = []
            selected_leaf_keys = set()
            # Track in-flight edges within this batch to implement virtual loss
            inflight_counts: Dict[Node, int] = {}
            
            # 1. Select leaves to expand
            for i in range(batch_size):
                # Try to pick distinct leaves within a batch to improve diversity
                attempts = 0
                while True:
                    node, path, leaf_board = self._select(
                        board.copy(), root, inflight_counts=inflight_counts, base_ply=(ply or 0)
                    )
                    leaf_key = leaf_board._transposition_key()
                    if leaf_board.is_game_over() or (leaf_key not in selected_leaf_keys) or attempts >= 2:
                        selected_leaf_keys.add(leaf_key)
                        break
                    attempts += 1
                if leaf_board.is_game_over():
                    v_leaf = self._terminal_value(leaf_board)
                    self._backpropagate(path, v_leaf)
                    sims_run_this_turn += 1
                else:
                    leaves.append((node, path, leaf_board))

            if not leaves:
                continue

            # 2. Batch evaluate leaves
            leaf_boards = [l[2] for l in leaves]
            p_list, v_list = self._infer_batch(leaf_boards)

            # 3. Expand and backpropagate
            for i, (node, path, leaf_board) in enumerate(leaves):
                p_logits, v_leaf = p_list[i], v_list[i]
                if not node.is_expanded():
                    node.expand(leaf_board, p_logits, encoder=self._enc, legal_only=self.cfg.legal_softmax)
                    self._prune_children(node)
                    self._register_children_in_tt(node, leaf_board)
                
                self._backpropagate(path, v_leaf)
                sims_run_this_turn += 1

        # Preserve root.n from backpropagations; do not overwrite with sims_run_this_turn
        
        # Debug: Log what happened during MCTS (quiet)
        if sims_run_this_turn == 0:
            logger.debug(f"MCTS ran {sims_to_run} simulations but no simulations completed! root_children={len(root.children)}")
        
        # Properly handle unvisited root children without corrupting the tree
        # Only update children that were actually visited during backpropagation
        for child in root.children.values():
            if child.n == 0:
                # Mark as unvisited but don't corrupt the tree with fake visits
                child.n = 0
                child.w = 0.0
                child.q = 0.0
        
        visit_counts: Dict[chess.Move, int] = {m: c.n for m, c in root.children.items()}
        policy = self._policy_from_root(root, board)
        self._last_sims_run = sims_run_this_turn
        # Expose last root for external inspection (e.g., debug prints)
        self._last_root = root
        return visit_counts, policy, float(v)

    def _prune_children(self, node: "Node") -> None:
        """Prune low-prior children to limit branching, if configured.

        - Keep only top-K by prior if max_children > 0
        - Drop children with prior < min_child_prior
        """
        try:
            if not node.children:
                return
            items = list(node.children.items())
            # Drop by min prior first
            if self.cfg.min_child_prior > 0.0:
                items = [(m, c) for (m, c) in items if float(c.prior) >= float(self.cfg.min_child_prior)]
            # Keep top-K by prior
            if self.cfg.max_children and self.cfg.max_children > 0 and len(items) > self.cfg.max_children:
                items.sort(key=lambda mc: float(mc[1].prior), reverse=True)
                items = items[: int(self.cfg.max_children)]
            # Rebuild dict if changed
            node.children = {m: c for (m, c) in items}
        except Exception:
            pass

    def _policy_from_root(self, root: Node, board: chess.Board) -> np.ndarray:
        pi = np.zeros(4672, dtype=np.float32)
        total = sum(c.n for c in root.children.values())
        if total > 0:
            for m, child in root.children.items():
                idx = move_to_index(board, m)
                pi[idx] = child.n / total
        else:
            # If no visits, use uniform distribution over legal moves
            legal_moves = list(board.legal_moves)
            if legal_moves:
                uniform_prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    idx = move_to_index(board, move)
                    pi[idx] = uniform_prob
        return pi

    def _select(self, board: chess.Board, root: Node, inflight_counts: Optional[Dict[Node, int]] = None, base_ply: int = 0) -> Tuple[Node, List[Node], chess.Board]:
        node = root
        path = [root]
        
        while node.is_expanded():
            if not node.children: # Terminal node
                break

            parent_visits = max(1, node.n)
            
            best_score = -1e9
            best_child = None
            
            for child in node.children.values():
                if child.n == 0:
                    # FPU scaled by parent Q to emphasize uncertain branches
                    fpu_bias = (self.cfg.fpu * (0.5 - node.q)) if self.cfg.parent_q_init else 0.0
                    q = -fpu_bias
                else:
                    q = child.q

                # Effective cpuct can be scheduled by ply (linear)
                depth = max(0, len(path) - 1)
                eff_cpuct = self._cpuct_at(base_ply + depth)
                u = eff_cpuct * child.prior * (math.sqrt(parent_visits) / (1.0 + child.n))
                score = q + u
                # Avoid instant backtracking at shallow depths if configured
                if self.cfg.no_instant_backtrack and len(path) >= 2 and child.move is not None and path[-1].move is not None:
                    prev = path[-1].move
                    mv = child.move
                    if mv.from_square == prev.to_square and mv.to_square == prev.from_square:
                        score -= 0.01
                # Apply virtual loss penalty if this edge is in-flight in current batch
                if inflight_counts is not None and self.cfg.virtual_loss > 0.0:
                    score -= float(inflight_counts.get(child, 0)) * float(self.cfg.virtual_loss)
                
                # Always add small jitter to break ties when UCB scores are identical
                if self.cfg.selection_jitter > 0:
                    score += (random.random() - 0.5) * self.cfg.selection_jitter
                else:
                    # Add minimal jitter even when selection_jitter is 0 to break ties
                    score += (random.random() - 0.5) * 0.001

                # Debug: Log UCB scores only when selection fails (reduced spam)

                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None:
                # Debug: Log why no child was selected
                logger.warning(f"No child selected from {len(node.children)} children")
                for move, child in node.children.items():
                    logger.warning(f"  {move}: prior={child.prior:.4f}, n={child.n}, q={child.q:.4f}")
                
                # Fallback: select first child if all UCB scores are identical
                if node.children:
                    best_child = next(iter(node.children.values()))
                    logger.warning(f"Fallback: selected first child {best_child.move}")
                else:
                    break # Should not happen if children exist
            
            board.push(best_child.move)
            node = self._tt_get(board._transposition_key()) or best_child
            path.append(node)
            # Mark this edge as in-flight for the batch to diversify subsequent selections
            if inflight_counts is not None:
                inflight_counts[best_child] = inflight_counts.get(best_child, 0) + 1

        return node, path, board

    def _cpuct_at(self, ply: int) -> float:
        try:
            start = self.cfg.cpuct_start
            end = self.cfg.cpuct_end
            span = int(self.cfg.cpuct_plies)
            if start is None or end is None or span <= 0:
                return float(self.cfg.cpuct)
            t = min(max(ply, 0), span) / float(span)
            return float(start) + (float(end) - float(start)) * t
        except Exception:
            return float(self.cfg.cpuct)

    def _backpropagate(self, path: List[Node], value: float) -> None:
        v = max(-1.0, min(1.0, float(value)))
        for node in reversed(path):
            node.n += 1
            node.w += v
            node.q = node.w / node.n
            v = -v

    def _add_dirichlet(self, root: Node) -> None:
        if not root.children or self.cfg.dirichlet_frac <= 0:
            return
        
        alpha = self.cfg.dirichlet_alpha
        frac = self.cfg.dirichlet_frac
        noise = np.random.dirichlet([alpha] * len(root.children))
        
        for i, child in enumerate(root.children.values()):
            child.prior = child.prior * (1 - frac) + noise[i] * frac

    @torch.no_grad()
    def _infer(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        key = board._transposition_key()
        cached = self.nn_cache.get(key)
        if cached:
            return cached
        
        arr = encode_board(board)
        if self.inference_backend is not None:
            p_b, v_b = self.inference_backend.infer_np(arr)
            p_np = p_b[0]
            v_np = float(v_b[0])
            # Adjust value orientation if model outputs from White's perspective
            if self.cfg.value_from_white and board.turn == chess.BLACK:
                v_np = -v_np
            out = (p_np, v_np)
            self.nn_cache.put(key, out)
            return out
        else:
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            try:
                x = x.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass
            self.model.eval()

            device_type = self.device.split(':')[0]
            use_amp = device_type in ("cuda", "mps")
            with torch.autocast(device_type=device_type, enabled=use_amp):
                p, v = self.model(x, return_ssl=False)  # Don't need SSL output for inference

        # Convert outputs to float32 numpy arrays
        p_np = p[0].detach().cpu().to(torch.float32).numpy() if self.inference_backend is None else p_np
        v_np = float(v.item()) if self.inference_backend is None else v_np
        # Adjust value orientation if model outputs from White's perspective
        if self.inference_backend is None and self.cfg.value_from_white and board.turn == chess.BLACK:
            v_np = -v_np
        
        # Validate outputs and handle invalid cases properly
        if np.any(np.isnan(p_np)) or np.any(np.isinf(p_np)):
            # Use proper fallback based on move priors instead of uniform distribution
            p_np = self._generate_prior_based_policy(board)
        
        if np.isnan(v_np) or np.isinf(v_np):
            # Use conservative value estimate for invalid outputs
            v_np = 0.0
        
        out = (p_np, v_np)
        self.nn_cache.put(key, out)
        return out

    @torch.no_grad()
    def _infer_batch(self, boards: List[chess.Board]) -> Tuple[List[np.ndarray], List[float]]:
        p_out: List[Optional[np.ndarray]] = [None] * len(boards)
        v_out: List[Optional[float]] = [None] * len(boards)
        to_eval_indices: List[int] = []
        to_eval_boards: List[chess.Board] = []

        for i, b in enumerate(boards):
            key = b._transposition_key()
            cached = self.nn_cache.get(key)
            if cached:
                p_logits, v = cached
                p_out[i] = p_logits
                v_out[i] = v
            else:
                to_eval_indices.append(i)
                to_eval_boards.append(b)
        
        if to_eval_boards:
            arr = np.stack([encode_board(b) for b in to_eval_boards], axis=0)
            if self.inference_backend is not None:
                p_np, v_np = self.inference_backend.infer_np(arr)
                # Flip values for black-to-move if needed
                if self.cfg.value_from_white:
                    for j, b in enumerate(to_eval_boards):
                        if b.turn == chess.BLACK:
                            v_np[j] = -float(v_np[j])
            else:
                x = torch.from_numpy(arr).to(self.device)
                try:
                    x = x.contiguous(memory_format=torch.channels_last)
                except Exception:
                    pass
                self.model.eval()

                device_type = self.device.split(':')[0]
                use_amp = device_type in ("cuda", "mps")
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    p, v = self.model(x, return_ssl=False)  # Don't need SSL output for inference

                p_np = p.detach().cpu().to(torch.float32).numpy()
                v_np = v.detach().cpu().to(torch.float32).numpy().flatten()
                if self.cfg.value_from_white:
                    for j, b in enumerate(to_eval_boards):
                        if b.turn == chess.BLACK:
                            v_np[j] = -float(v_np[j])
            
            for j, original_idx in enumerate(to_eval_indices):
                key = to_eval_boards[j]._transposition_key()
                result = (p_np[j], float(v_np[j]))
                p_out[original_idx] = result[0]
                v_out[original_idx] = result[1]
                self.nn_cache.put(key, result)
        
        return p_out, v_out

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        # Penalize draws slightly to encourage finding wins
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return float(self.cfg.draw_penalty)
        return 0.0

    def _tt_get(self, key: str) -> Optional[Node]:
        node = self.tt.get(key)
        if node:
            self.tt.move_to_end(key)
            self.tt_hits += 1
            return node
        self.tt_misses += 1
        return None
    
    def _get_position_hash(self, board: chess.Board) -> str:
        """Get stable position hash for transposition table."""
        # Use Zobrist hash for efficient position comparison
        return board._transposition_key()

    def _tt_put(self, key: str, node: Node) -> None:
        self.tt[key] = node
        self.simulations_run += 1
        # Defer cleanup when bulk-inserting to avoid dict-size-changed during iteration
        if not self._in_bulk_put:
            self._check_memory_pressure()
            if len(self.tt) > self.cfg.tt_capacity or self.simulations_run - self._last_cleanup > self.cfg.tt_cleanup_frequency:
                self._cleanup_tt()

    def _check_memory_pressure(self) -> bool:
        """Check if memory usage is high and trigger cleanup if needed."""
        if not psutil_available:
            return False
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self._memory_cleanup_threshold:
                logger.warning(
                    f"Memory pressure detected: {memory_percent:.1f}%, triggering aggressive cleanup"
                )
                self._cleanup_deep_branches(max_depth=20)  # More aggressive pruning
                return True
        except Exception as e:
            logger.warning(f"Could not check memory pressure: {e}")
        return False
    
    def _cleanup_tt(self) -> None:
        """Lightweight TT cleanup: favor LRU trimming; avoid deep recursion.

        Rationale: deep branch recursion can be very heavy under multiprocessing and
        large trees, causing workers to stall or appear unresponsive. We keep cleanup
        cheap and frequent: trim LRU entries to ~80% capacity and check memory pressure.
        """
        # Frequency-based and wall-clock based cleanup
        if (self.simulations_run - self._last_cleanup < 500) and (time.time() - self._last_cleanup_wall < max(1, int(self.cfg.tt_cleanup_interval_s))):
            return
        self._last_cleanup = self.simulations_run
        self._last_cleanup_wall = time.time()
        # Trim LRU to 80% of capacity if exceeded
        target = int(self.cfg.tt_capacity * 0.8)
        if len(self.tt) > target:
            excess = len(self.tt) - target
            for _ in range(excess):
                try:
                    self.tt.popitem(last=False)
                except Exception:
                    break
        # Check memory pressure opportunistically
        self._check_memory_pressure()
    
    def _cleanup_deep_branches(self, max_depth: int = 30):
        """Recursively clean up deep branches to prevent memory explosion."""
        def cleanup_node(node: Node, depth: int):
            if depth > max_depth:
                # Prune this branch
                node.children.clear()
                return
            
            for child in node.children.values():
                cleanup_node(child, depth + 1)
        
        # Find root nodes and clean them
        root_hashes = set()
        for board_hash in self.tt:
            root_hashes.add(board_hash)
        
        for board_hash in root_hashes:
            if board_hash in self.tt:
                node = self.tt[board_hash]
                cleanup_node(node, 0)

    def _register_children_in_tt(self, node: Node, board: chess.Board) -> None:
        # Bulk insert: defer cleanup until after registering all children
        prev = self._in_bulk_put
        self._in_bulk_put = True
        try:
            for m, child in node.children.items():
                # Avoid copying move stack to reduce overhead
                try:
                    b2 = board.copy(stack=False)
                except TypeError:
                    # Fallback for older python-chess versions without stack parameter
                    b2 = board.copy()
                b2.push(m)
                self._tt_put(b2._transposition_key(), child)
        finally:
            self._in_bulk_put = prev
        # Do not run heavy cleanup here; periodic cleanup will be triggered by _tt_put
    
    def _generate_prior_based_policy(self, board: chess.Board) -> np.ndarray:
        """Generate fallback policy based on move priors when NN output is invalid."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return np.zeros(4672, dtype=np.float32)
        
        # Create policy based on move priors (piece values, position, etc.)
        policy = np.zeros(4672, dtype=np.float32)
        total_weight = 0.0
        
        for move in legal_moves:
            idx = move_to_index(board, move)
            # Weight based on piece value and move type
            weight = self._get_move_weight(board, move)
            policy[idx] = weight
            total_weight += weight
        
        # Normalize to probabilities
        if total_weight > 0 and np.isfinite(total_weight):
            policy /= float(total_weight)
        else:
            # Uniform over legal moves as ultimate fallback
            for move in legal_moves:
                policy[move_to_index(board, move)] = 1.0 / len(legal_moves)
        return policy.astype(np.float32, copy=False)
    
    def _get_move_weight(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate move weight based on piece value and move characteristics."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return 1.0
        
        # Base weight from piece value
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        base_weight = piece_values.get(piece.piece_type, 1.0)
        
        # Bonus for captures
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                base_weight += piece_values.get(captured.piece_type, 0)
        
        # Bonus for checks
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            base_weight *= 1.5
        
        return base_weight
