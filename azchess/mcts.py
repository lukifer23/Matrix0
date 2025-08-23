from __future__ import annotations

import math
import random
import time
import threading
import queue
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


def _ensure_contiguous_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Ensure tensor is contiguous, with debug logging for MPS compatibility."""
    if not tensor.is_contiguous():
        logger.debug(f"Making {name} contiguous (shape: {tensor.shape}, device: {tensor.device})")
        return tensor.contiguous()
    return tensor


def _ensure_contiguous_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Ensure numpy array is contiguous for MPS compatibility."""
    if not array.flags.c_contiguous:
        logger.debug(f"Making {name} contiguous (shape: {array.shape})")
        return np.ascontiguousarray(array)
    return array


class ThreadPool:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.threads = []
        self._shutdown_event = threading.Event()
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while not self._shutdown_event.is_set():
            try:
                task = self.tasks.get(timeout=1.0)  # 1 second timeout
                if task is None:
                    break
                func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    self.results.put(result)
                except Exception as e:
                    logger.error(f"Worker thread error: {e}")
                    self.results.put(e)
                finally:
                    self.tasks.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread critical error: {e}")
                break

    def add_task(self, func, *args, **kwargs):
        if not self._shutdown_event.is_set():
            self.tasks.put((func, args, kwargs))

    def get_results(self):
        try:
            return self.results.get(timeout=5.0)  # 5 second timeout
        except queue.Empty:
            return None

    def wait_completion(self, timeout=30.0):
        """Wait for completion with robust timeout and deadlock prevention."""
        start_time = time.time()
        max_wait = timeout
        
        while not self.tasks.empty():
            if time.time() - start_time > max_wait:
                remaining = self.tasks.qsize()
                logger.warning(f"ThreadPool timeout after {timeout}s, {remaining} tasks remaining")
                # Cancel remaining tasks to prevent deadlock
                self.cancel_pending_tasks()
                return False
            
            # Use shorter sleep intervals for better responsiveness
            time.sleep(0.001)  # 1ms sleep instead of 10ms
        
        return True
    
    def cancel_pending_tasks(self):
        """Cancel all pending tasks to prevent deadlocks."""
        try:
            # Clear the task queue
            while not self.tasks.empty():
                try:
                    self.tasks.get_nowait()
                except:
                    break
            logger.info("Cancelled pending ThreadPool tasks")
        except Exception as e:
            logger.error(f"Error cancelling ThreadPool tasks: {e}")



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
    # Memory management
    enable_memory_cleanup: bool = True  # Enable automatic memory cleanup
    memory_cleanup_threshold_mb: int = 1024  # Trigger cleanup at this memory usage
    max_tree_nodes: int = 100000     # Maximum nodes before forced cleanup
    # NEW: Multi-threading optimizations
    num_threads: int = 6             # Use all 6 CPU cores
    parallel_simulations: bool = True # Enable parallel MCTS simulations
    simulation_batch_size: int = 16   # Batch simulations for efficiency
    tree_parallelism: bool = True     # Enable tree-level parallelism

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

    def _expand(self, board: chess.Board, p_logits: np.ndarray, encoder: Optional[MoveEncoder] = None, legal_only: bool = False) -> None:
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
                sel = _ensure_contiguous_tensor(torch.from_numpy(logits[idxs]), "legal_logits")
                sel = torch.softmax(sel, dim=-1).numpy()
                probs = None
            else:
                probs = _ensure_contiguous_tensor(torch.from_numpy(logits), "policy_logits")
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
                # Ensure probs is contiguous after operations
                probs = _ensure_contiguous_array(probs, "noisy_probs")

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
    def __init__(
        self,
        cfg: MCTSConfig,
        model=None,
        device: str = "cpu",
        inference_backend=None,
        num_threads: int = None,
    ):
        if isinstance(cfg, torch.nn.Module) or cfg is None:
            cfg, model = model, cfg

        self.cfg = cfg
        self.device = device
        self.model = model
        self.inference_backend = inference_backend

        self._tt = {}
        self._tt_lock = threading.Lock()
        self._tt_cleanup_counter = 0
        self._last_cleanup_time = time.time()

        self.tt = OrderedDict()  # Transposition table
        self.nn_cache = LRUCache(10000)  # Neural network cache
        self.simulations_run = 0
        self._last_sims_run = 0
        self._last_root = None
        self._last_cleanup = 0
        self._in_bulk_put = False
        self._memory_cleanup_threshold = 85.0
        self.tt_hits = 0
        self.tt_misses = 0

        # Move encoder cache
        self._enc: Optional[MoveEncoder] = (
            MoveEncoder() if bool(cfg.encoder_cache) else None
        )
        self._last_cleanup_wall: float = time.time()

        # Configure threading
        self.num_threads = (
            num_threads if num_threads is not None else getattr(cfg, "num_threads", 1)
        )
        if self.num_threads > 1:
            self.thread_pool = ThreadPool(self.num_threads)
            logger.info(
                f"MCTS initialized with {self.num_threads} threads for parallel simulation"
            )
        else:
            self.thread_pool = None
            logger.info("MCTS running in single-threaded mode")

        self.lock = threading.Lock()

    @torch.no_grad()
    def run(self, board: chess.Board, num_simulations: Optional[int] = None, ply: Optional[int] = None) -> Tuple[Dict[chess.Move, int], np.ndarray, float]:
        """Run MCTS with safety timeout to prevent deadlocks."""
        start_time = time.time()
        max_runtime = 120.0  # Maximum 2 minutes per MCTS run
        
        try:
            if board.is_game_over():
                return {}, np.zeros(4672, dtype=np.float32), self._terminal_value(board)

            key = board._transposition_key()
            root = self._tt_get(key)
            if root is None:
                root = Node()
                p_logits, v = self._infer(board)
                root._expand(board, p_logits, encoder=self._enc, legal_only=self.cfg.legal_softmax)
                self._prune_children(root)
                self._tt_put(key, root, skip_lock=True)
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

            # Periodic memory cleanup
            if getattr(self.cfg, 'enable_memory_cleanup', True):
                self._cleanup_memory()

            # Check runtime before starting simulations
            if time.time() - start_time > max_runtime * 0.5:
                logger.warning("MCTS setup took too long, reducing simulations")
                sims_to_run = max(50, sims_to_run // 2)  # Minimum 50 sims for quality

            # Enhanced parallelization with batched inference for better throughput
            # FIXED: Robust parallel MCTS with proper error handling and deadlock prevention
            if self.num_threads > 1 and self.thread_pool is not None:
                try:
                    # Use the new batched inference approach
                    self._run_simulations_parallel_batched(board, root, sims_to_run)
                except Exception as e:
                    logger.error(f"Batched MCTS failed: {e}, falling back to single-threaded")
                    # Robust fallback: clear thread pool and run single-threaded
                    try:
                        if self.thread_pool:
                            self.thread_pool.cancel_pending_tasks()
                    except:
                        pass
                    # Run remaining simulations single-threaded
                    for _ in range(sims_to_run):
                        self._run_simulation(board, root)
            else:
                # Single-threaded MCTS (reliable fallback)
                for _ in range(sims_to_run):
                    self._run_simulation(board, root)

            visit_counts: Dict[chess.Move, int] = {m: c.n for m, c in root.children.items()}
            policy = self._policy_from_root(root, board)
            self._last_sims_run = sims_to_run
            # Expose last root for external inspection (e.g., debug prints)
            self._last_root = root
            
            runtime = time.time() - start_time
            if runtime > max_runtime * 0.8:
                logger.warning(f"MCTS run took {runtime:.2f}s (close to {max_runtime}s limit)")
                
            return visit_counts, policy, float(v)
            
        except Exception as e:
            logger.error(f"MCTS run error: {e}")
            # Return safe fallback values
            return {}, np.zeros(4672, dtype=np.float32), 0.0

    def _run_simulations_parallel_batched(self, board: chess.Board, root: Node, num_simulations: int):
        """Run simulations in parallel with simple batched inference."""
        # Simple approach: collect all positions that need inference, then batch them
        
        # First, run all simulations to collect leaf positions
        leaf_positions = []
        leaf_nodes = []
        
        # Run simulations in parallel to collect positions
        for _ in range(num_simulations):
            self.thread_pool.add_task(self._collect_leaf_position, board.copy(), root, leaf_positions, leaf_nodes)
        
        # Wait for all simulations to complete
        if not self.thread_pool.wait_completion(timeout=30.0):
            logger.warning("Position collection timed out")
            return
        
        # Now do batched inference on all collected positions
        if leaf_positions:
            try:
                # Encode all positions
                encoded_batch = []
                for pos in leaf_positions:
                    if pos is not None:
                        encoded = encode_board(pos)
                        if encoded is not None:
                            encoded_batch.append(encoded)
                
                if encoded_batch:
                    # Limit batch size to what shared memory can handle (typically 32)
                    max_batch_size = 32  # Shared memory limit
                    if len(encoded_batch) > max_batch_size:
                        logger.debug(f"Limiting batch from {len(encoded_batch)} to {max_batch_size} positions")
                        encoded_batch = encoded_batch[:max_batch_size]
                        leaf_nodes = leaf_nodes[:max_batch_size]
                        leaf_positions = leaf_positions[:max_batch_size]
                    
                    # Stack into batch tensor
                    batch_tensor = np.stack(encoded_batch, axis=0)
                    logger.debug(f"Running batched inference on {len(encoded_batch)} positions")
                    
                    # Use inference client for batched inference
                    if hasattr(self, 'inference_backend') and self.inference_backend is not None:
                        policies, values = self.inference_backend.infer_np(batch_tensor)
                        
                        # Apply results back to nodes
                        for i, (node, policy, value) in enumerate(zip(leaf_nodes, policies, values)):
                            if node is not None and not node.is_expanded():
                                node._expand(leaf_positions[i], policy, encoder=self._enc, legal_only=self.cfg.legal_softmax)
                                self._prune_children(node)
                                self._register_children_in_tt(node, leaf_positions[i], skip_lock=True)
                    else:
                        # Fallback to individual inference
                        for i, pos in enumerate(leaf_positions):
                            if pos is not None:
                                try:
                                    p_logits, v = self._infer(pos)
                                    node = leaf_nodes[i]
                                    if node is not None and not node.is_expanded():
                                        node._expand(pos, p_logits, encoder=self._enc, legal_only=self.cfg.legal_softmax)
                                        self._prune_children(node)
                                        self._register_children_in_tt(node, pos, skip_lock=True)
                                except Exception as e:
                                    logger.error(f"Individual inference failed: {e}")
                                    
            except Exception as e:
                logger.error(f"Batched inference failed: {e}, falling back to individual")
                # Fallback to individual inference
                for pos in leaf_positions:
                    if pos is not None:
                        try:
                            self._infer(pos)
                        except:
                            pass

    def _collect_leaf_position(self, board, root, leaf_positions, leaf_nodes):
        """Collect the leaf position from a simulation."""
        try:
            node, path, leaf_board = self._select(board, root)
            
            if leaf_board.is_game_over():
                v_leaf = self._terminal_value(leaf_board)
            else:
                # Store position and node for later batched inference
                leaf_positions.append(leaf_board)
                leaf_nodes.append(node)
                v_leaf = 0.0  # Will be updated after batched inference
            
            self._backpropagate(path, v_leaf)
            
        except Exception as e:
            logger.error(f"Position collection error: {e}")
            leaf_positions.append(None)
            leaf_nodes.append(None)

    def _run_simulation(self, board: chess.Board, root: Node):
        """Run a single MCTS simulation with robust error handling."""
        try:
            # CRITICAL FIX: Clone the board to prevent corruption across parallel simulations
            simulation_board = board.copy()
            
            node, path, leaf_board = self._select(simulation_board, root)
            
            if leaf_board.is_game_over():
                v_leaf = self._terminal_value(leaf_board)
            else:
                try:
                    p_logits, v_leaf = self._infer(leaf_board)
                    if not node.is_expanded():
                        node._expand(leaf_board, p_logits, encoder=self._enc, legal_only=self.cfg.legal_softmax)
                        self._prune_children(node)
                        self._register_children_in_tt(node, leaf_board, skip_lock=True)
                except Exception as e:
                    logger.error(f"Inference error in simulation: {e}")
                    # Use neutral value on inference failure
                    v_leaf = 0.0

            self._backpropagate(path, v_leaf)
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            # Don't let simulation errors crash the entire MCTS
            # The task will still be marked as done by the worker thread

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
            with self.lock:
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
        with self.lock:
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
        """Get policy and value from the neural network with robust error handling."""
        try:
            # Encode board position using the correct function
            encoded = encode_board(board)
            if encoded is None:
                logger.warning("Failed to encode board, using fallback values")
                return np.ones(4672, dtype=np.float32) / 4672, 0.0
            
            # Ensure tensor is contiguous for MPS compatibility
            encoded = _ensure_contiguous_array(encoded)
            
            # Add batch dimension if needed
            if encoded.ndim == 3:
                encoded = np.expand_dims(encoded, 0)
            
            # Inference with timeout and retry
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    if hasattr(self, 'inference_backend') and self.inference_backend is not None:
                        # Use inference client for multi-threaded MCTS
                        policy, value = self.inference_backend.infer_np(encoded)
                    else:
                        # Direct inference for single-threaded
                        with torch.no_grad():
                            policy_logits, value_tensor = self.model(torch.from_numpy(encoded).to(self.device))
                            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                            value = value_tensor.cpu().numpy().flatten()
                    
                    # Validate outputs
                    if policy.shape[0] != encoded.shape[0] or value.shape[0] != encoded.shape[0]:
                        raise ValueError(f"Output shape mismatch: policy={policy.shape}, value={value.shape}, input={encoded.shape}")
                    
                    return policy[0], float(value[0])
                    
                except TimeoutError as e:
                    if attempt < max_retries:
                        logger.warning(f"Inference timeout (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"Inference timeout after {max_retries + 1} attempts: {e}")
                        # Return fallback values to prevent MCTS crash
                        return np.ones(4672, dtype=np.float32) / 4672, 0.0
                        
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Inference error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(0.1)
                        continue
                    else:
                        logger.error(f"Inference failed after {max_retries + 1} attempts: {e}")
                        # Return fallback values to prevent MCTS crash
                        return np.ones(4672, dtype=np.float32) / 4672, 0.0
                        
        except Exception as e:
            logger.error(f"Critical inference error: {e}")
            # Return safe fallback values
            return np.ones(4672, dtype=np.float32) / 4672, 0.0

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        # Penalize draws slightly to encourage finding wins
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return float(self.cfg.draw_penalty)
        return 0.0

    def _tt_get(self, key: str) -> Optional[Node]:
        with self.lock:
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

    def _tt_put(self, key: str, node: Node, skip_lock: bool = False) -> None:
        """Put a node in the transposition table.
        
        Args:
            key: Position hash key
            node: Node to store
            skip_lock: If True, skip lock acquisition (for when called from locked context)
        """
        def _do_put():
            self.tt[key] = node
            self.simulations_run += 1
            # Defer cleanup when bulk-inserting to avoid dict-size-changed during iteration
            if not self._in_bulk_put:
                self._check_memory_pressure()
                if len(self.tt) > self.cfg.tt_capacity or self.simulations_run - self._last_cleanup > self.cfg.tt_cleanup_frequency:
                    self._cleanup_tt()
        
        if skip_lock:
            _do_put()
        else:
            with self.lock:
                _do_put()

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
                self._cleanup_deep_branches(max_depth=10)  # More aggressive pruning
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

    def _register_children_in_tt(self, node: Node, board: chess.Board, skip_lock: bool = False) -> None:
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
                self._tt_put(b2._transposition_key(), child, skip_lock=skip_lock)
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

    def _cleanup_memory(self, force: bool = False) -> None:
        """Clean up memory by clearing caches and old TT entries."""
        current_time = time.time()

        # Only cleanup periodically unless forced
        if not force and current_time - self._last_cleanup_wall < self.cfg.tt_cleanup_interval_s:
            return

        try:
            # Get current memory usage
            if psutil_available:
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_threshold = getattr(self.cfg, 'memory_cleanup_threshold_mb', 2048)

                # Force cleanup if memory usage is too high
                if memory_mb > memory_threshold:
                    logger.info(f"Memory usage {memory_mb:.0f}MB exceeds threshold {memory_threshold}MB, forcing cleanup")
                    force = True
        except Exception:
            pass

        if force:
            # Aggressive cleanup
            logger.debug("Performing aggressive memory cleanup")

            # Clear NN cache
            self.nn_cache.cache.clear()

            # Clear old TT entries (keep only recent ones)
            if len(self.tt) > 1000:
                # Keep only the 1000 most recently accessed entries
                items = list(self.tt.items())
                items.sort(key=lambda x: getattr(x[1], '_last_access', 0), reverse=True)
                self.tt.clear()
                for key, node in items[:1000]:
                    self.tt[key] = node

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_cleanup_wall = current_time

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        stats = {
            'tt_entries': len(self.tt),
            'nn_cache_entries': len(self.nn_cache.cache),
            'simulations_run': self.simulations_run
        }

        if psutil_available:
            try:
                process = psutil.Process()
                stats['memory_mb'] = process.memory_info().rss // (1024 * 1024)
            except Exception:
                stats['memory_mb'] = 0

        return stats

    def reset(self):
        """Reset the MCTS instance, clearing all caches."""
        self.tt.clear()
        self.nn_cache.cache.clear()
        self.simulations_run = 0
        self._last_sims_run = 0
        self._last_root = None
        self.tt_hits = 0
        self.tt_misses = 0

        # Force cleanup
        self._cleanup_memory(force=True)
