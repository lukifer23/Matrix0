from __future__ import annotations

from multiprocessing import Event
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
import threading

import numpy as np
import torch

from ..model import PolicyValueNet

# --- Shared Memory IPC Implementation ---


def setup_shared_memory_for_worker(
    worker_id: int, planes: int, policy_size: int, max_batch_size: int
) -> Dict[str, Any]:
    """Creates shared memory tensors and events for a single worker."""
    return {
        "request_tensor": torch.zeros(
            (max_batch_size, planes, 8, 8), dtype=torch.float32
        ).share_memory_(),
        "response_policy_tensor": torch.zeros(
            (max_batch_size, policy_size), dtype=torch.float32
        ).share_memory_(),
        "response_value_tensor": torch.zeros(
            (max_batch_size, 1), dtype=torch.float32
        ).share_memory_(),
        "request_event": Event(),
        "response_event": Event(),
        "batch_size_tensor": torch.tensor([0], dtype=torch.int32).share_memory_(),
    }


def _validate_events(events: List[Event]) -> bool:
    """Validate that all events are in a usable state."""
    try:
        for i, ev in enumerate(events):
            # Test if event can be checked without error
            _ = ev.is_set()
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _recreate_worker_events(shared_memory_resources: List[Dict[str, Any]], attempt: int) -> None:
    """Recreate all worker events to fix corruption."""
    logger = logging.getLogger(__name__)
    logger.warning(f"Recreating worker events (attempt {attempt})")
    
    for i, res in enumerate(shared_memory_resources):
        try:
            res["request_event"] = Event()
            res["response_event"] = Event()
            logger.debug(f"Recreated events for worker {i}")
        except Exception as recreate_error:
            logger.error(f"Failed to recreate events for worker {i}: {recreate_error}")


def run_inference_server(
    device: str,
    model_cfg: dict,
    model_state_dict: Optional[Dict[str, Any]],
    stop_event: Any,
    server_ready_event: Any,
    shared_memory_resources: List[Dict[str, Any]],
):
    """Inference server using shared memory for communication."""
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Inference server starting on device: {device}")
        logger.info(f"Available workers: {len(shared_memory_resources)}")

        try:
            logger.info(f"Creating model from config: {model_cfg}")
            model_creation_start = time.time()
            model = PolicyValueNet.from_config(model_cfg)
            model_creation_time = time.time() - model_creation_start
            logger.info(f"Model created successfully in {model_creation_time:.3f}s, moving to device...")

            # Clear MPS cache before moving model
            if device == "mps":
                logger.info("Clearing MPS cache before model placement")
                import torch.mps
                torch.mps.empty_cache()

            # Move model to device with error handling
            model_move_start = time.time()
            model = model.to(device)
            model_move_time = time.time() - model_move_start
            logger.info(f"Model successfully moved to device in {model_move_time:.3f}s: {device}")

        except Exception as e:
            logger.error(f"Failed to create or move model to device: {e}")
            logger.error(f"Model config: {model_cfg}")
            logger.error(f"Target device: {device}")
            raise
        logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )

        if model_state_dict:
            logger.info(
                f"Loading model from state_dict with {len(model_state_dict)} layers"
            )
            try:
                # Clear cache before loading state dict
                if device == "mps":
                    torch.mps.empty_cache()

                missing, unexpected = model.load_state_dict(
                    model_state_dict, strict=False
                )
                if missing:
                    total_expected = len(model_state_dict) + len(missing)
                    logger.warning(
                        f"Missing keys during load (initialized from defaults): {len(missing)}/{total_expected} keys "
                        f"({len(model_state_dict)} loaded successfully)"
                    )
                    logger.warning(f"Missing keys: {sorted(list(missing))}")
                    logger.debug(f"Successfully loaded keys: {sorted(list(model_state_dict.keys()))}")
                if unexpected:
                    logger.warning(
                        f"Unexpected keys during load (ignored): {len(unexpected)} keys"
                    )
                    logger.debug(f"Unexpected keys: {sorted(list(unexpected))[:10]}")  # Show more keys for debugging
                logger.info("Model loaded from state_dict successfully (non-strict).")

                # Log detailed parameter information
                actual_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Model loaded with {actual_params:,} total parameters")

                # If we have missing keys, log a summary but don't fail
                if len(missing) > 10:
                    logger.warning(f"High number of missing keys ({len(missing)}), model may not perform as expected")

            except Exception as e:
                logger.error(
                    f"Model state_dict load failed: {e}; attempting with individual key matching."
                )
                # Try to load individual keys that match
                try:
                    model_dict = model.state_dict()
                    matched_keys = []
                    for key, value in model_state_dict.items():
                        if key in model_dict and model_dict[key].shape == value.shape:
                            model_dict[key] = value
                            matched_keys.append(key)
                        elif key in model_dict:
                            logger.debug(f"Shape mismatch for {key}: model {model_dict[key].shape} vs checkpoint {value.shape}")

                    model.load_state_dict(model_dict, strict=False)
                    logger.info(f"Loaded {len(matched_keys)} keys successfully with manual matching")
                    if len(matched_keys) < len(model_dict) * 0.5:  # Less than 50% keys loaded
                        logger.error(f"Only {len(matched_keys)}/{len(model_dict)} keys matched - model may not work properly")
                except Exception as fallback_error:
                    logger.error(f"All loading attempts failed: {fallback_error}")
                    raise
        else:
            logger.warning("No model state_dict provided, using random weights.")

        # Log device and memory information
        logger.info(f"Model device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            logger.info(
                f"CUDA memory: {torch.cuda.memory_allocated()/1024/1024:.1f}MB allocated"
            )
        elif device == "mps":
            try:
                import torch.mps
                logger.info(
                    f"MPS memory info: available={torch.mps.is_available()}, "
                    f"built={torch.mps.is_built()}"
                )
            except Exception as e:
                logger.debug(f"Could not get MPS memory info: {e}")

        server_ready_event.set()
        logger.info("Inference server ready")

        # Add heartbeat logging to monitor server health
        server_start_time = time.time()
        heartbeat_counter = 0
        last_heartbeat = time.time()
        heartbeat_interval = 30.0  # Log heartbeat every 30 seconds

        device_type = device.split(":")[0]
        use_amp = device_type in ("cuda", "mps")

        worker_events = [res["request_event"] for res in shared_memory_resources]
        event_to_worker = {ev: i for i, ev in enumerate(worker_events)}
        timeout = None

        logger.debug("Inference server entering main processing loop.")
        event_recreation_count = 0
        max_event_recreations = 3

        while not stop_event.is_set():
            try:
                # CRITICAL FIX: Validate events before use to prevent corruption
                if not _validate_events(worker_events):
                    logger.warning("Detected corrupted events, recreating them")
                    if event_recreation_count < max_event_recreations:
                        event_recreation_count += 1
                        _recreate_worker_events(shared_memory_resources, event_recreation_count)
                        worker_events = [res["request_event"] for res in shared_memory_resources]
                        event_to_worker = {ev: i for i, ev in enumerate(worker_events)}
                        time.sleep(0.1)
                        continue
                    else:
                        logger.error("Max event recreations reached, stopping inference server")
                        break

                # Add heartbeat logging
                if time.time() - last_heartbeat > heartbeat_interval:
                    logger.info(f"Inference server heartbeat - uptime: {time.time() - server_start_time:.1f}s, requests: {heartbeat_counter}")
                    last_heartbeat = time.time()
                    heartbeat_counter += 1

                # Custom wait implementation for multiprocessing.Event compatibility
                ready_events = []
                wait_start_time = time.time()

                while not stop_event.is_set():
                    current_time = time.time()
                    if timeout is not None and (current_time - wait_start_time) > timeout:
                        break

                    ready_events = []
                    for i, ev in enumerate(worker_events):
                        try:
                            if ev.is_set():
                                ready_events.append(ev)
                                # Clear the event for next use
                                ev.clear()
                        except (OSError, ValueError) as e:
                            # Event is corrupted
                            logger.warning(f"Event {i} is corrupted: {e}")
                            if event_recreation_count < max_event_recreations:
                                event_recreation_count += 1
                                _recreate_worker_events(shared_memory_resources, event_recreation_count)
                                worker_events = [res["request_event"] for res in shared_memory_resources]
                                event_to_worker = {ev: i for i, ev in enumerate(worker_events)}
                                break
                            else:
                                logger.error("Max event recreations reached, stopping inference server")
                                return

                    if stop_event.is_set():
                        break

                    if ready_events:
                        break

                    # Small sleep to avoid busy waiting
                    time.sleep(0.001)

                if stop_event.is_set():
                    break
                if not ready_events:
                    continue
                batch_indices = [event_to_worker[ev] for ev in ready_events]
                logger.debug(
                    f"Inference server processing batch from workers: {batch_indices}"
                )
            except Exception as e:
                logger.error(f"Error in inference server main loop: {e}")
                if event_recreation_count < max_event_recreations:
                    event_recreation_count += 1
                    logger.warning(f"Event objects may be invalid, recreating them (attempt {event_recreation_count}/{max_event_recreations})")

                    # Recreate events for all workers
                    for i, res in enumerate(shared_memory_resources):
                        try:
                            res["request_event"] = Event()
                            res["response_event"] = Event()
                            logger.debug(f"Recreated events for worker {i}")
                        except Exception as recreate_error:
                            logger.error(f"Failed to recreate events for worker {i}: {recreate_error}")

                    # Update worker_events and event_to_worker mapping
                    worker_events = [res["request_event"] for res in shared_memory_resources]
                    event_to_worker = {ev: i for i, ev in enumerate(worker_events)}

                    # Brief pause before continuing
                    time.sleep(0.1)
                    continue
                else:
                    logger.error(f"Max event recreations reached, stopping inference server")
                    break
            except Exception as e:
                logger.error(f"Unexpected error in inference server main loop: {e}")
                continue

            # Prepare batch from shared memory with better batching logic
            tensors_to_process = []
            batch_sizes = {}
            total_batch_size = 0

            try:
                for worker_id in batch_indices:
                    res = shared_memory_resources[worker_id]
                    batch_size = res["batch_size_tensor"].item()
                    logger.debug(f"Worker {worker_id}: batch_size={batch_size}")
                    if batch_size <= 0:
                        # Spurious wakeup or protocol error; ignore
                        logger.debug(
                            f"Worker {worker_id}: skipping due to batch_size <= 0"
                        )
                        continue
                    # Clamp to max capacity to avoid overflow
                    max_bs = res["request_tensor"].shape[0]
                    if batch_size > max_bs:
                        logger.warning(
                            f"Worker {worker_id} batch_size {batch_size} > capacity {max_bs}; clamping"
                        )
                        batch_size = max_bs
                        res["batch_size_tensor"][0] = max_bs

                    # Only add to batch if we have reasonable size (avoid tiny batches)
                    # OPTIMIZATION: Accumulate larger batches for better GPU utilization
                    if (
                        total_batch_size == 0 or total_batch_size < 16
                    ):  # Target 16+ samples per batch
                        try:
                            tensor_slice = res["request_tensor"][:batch_size]
                            logger.debug(f"Worker {worker_id}: extracting tensor slice of shape {tensor_slice.shape}")
                            tensors_to_process.append(tensor_slice)
                            batch_sizes[worker_id] = batch_size
                            total_batch_size += batch_size
                            res["request_event"].clear()  # Clear event after reading
                            logger.debug(
                                f"Accumulating batch: worker {worker_id}, size {batch_size}, total {total_batch_size}"
                            )
                        except Exception as slice_error:
                            logger.error(f"Failed to extract tensor slice for worker {worker_id}: {slice_error}")
                            logger.error(f"Request tensor shape: {res['request_tensor'].shape}, batch_size: {batch_size}")
                            continue
                    else:
                        # Process batches immediately to avoid timeouts
                        try:
                            tensor_slice = res["request_tensor"][:batch_size]
                            logger.debug(f"Worker {worker_id}: extracting tensor slice of shape {tensor_slice.shape}")
                            tensors_to_process.append(tensor_slice)
                            batch_sizes[worker_id] = batch_size
                            total_batch_size += batch_size
                            res["request_event"].clear()
                            logger.debug(
                                f"Processing batch immediately: worker {worker_id}, size {batch_size}, total {total_batch_size}"
                            )
                        except Exception as slice_error:
                            logger.error(f"Failed to extract tensor slice for worker {worker_id}: {slice_error}")
                            logger.error(f"Request tensor shape: {res['request_tensor'].shape}, batch_size: {batch_size}")
                            continue

                if not tensors_to_process:
                    logger.debug("No tensors to process, continuing...")
                    continue

                # Process larger batches for better GPU utilization
                try:
                    batch_tensor = torch.cat(tensors_to_process, dim=0).to(device)
                    logger.info(f"Processing batch of size {batch_tensor.shape[0]} from {len(batch_sizes)} workers")
                    logger.debug(f"Batch tensor shape: {batch_tensor.shape}, device: {batch_tensor.device}")
                except Exception as cat_error:
                    logger.error(f"Failed to concatenate tensors: {cat_error}")
                    logger.error(f"Number of tensors: {len(tensors_to_process)}")
                    for i, tensor in enumerate(tensors_to_process):
                        logger.error(f"Tensor {i} shape: {tensor.shape}, device: {tensor.device}")
                    continue

                # MPS OPTIMIZATION: Use memory format optimization for better performance
                if device_type == "mps":
                    try:
                        # Use channels_last memory format for better MPS performance
                        batch_tensor = batch_tensor.contiguous(
                            memory_format=torch.channels_last
                        )
                    except Exception:
                        # Fallback to default if channels_last not supported
                        pass

                # Inference with mixed precision if available
                inference_start = time.time()
                with torch.no_grad():
                    try:
                        logger.info(f"Starting inference for batch size {batch_tensor.shape[0]} on {device_type}")
                        model_start = time.time()

                        if use_amp:
                            with torch.autocast(device_type=device_type):
                                policy_logits, value_tensor = model(batch_tensor)
                        else:
                            policy_logits, value_tensor = model(batch_tensor)

                        model_time = time.time() - model_start
                        logger.info(f"Model forward pass completed in {model_time:.3f}s")

                        # Validate model outputs
                        if policy_logits is None or value_tensor is None:
                            raise ValueError("Model returned None outputs")

                        inference_time = time.time() - inference_start
                        logger.info(f"Full inference completed in {inference_time:.3f}s for batch size {batch_tensor.shape[0]}")

                        # Convert to numpy for response with strict dtype/shape
                        policy = torch.softmax(policy_logits, dim=-1).detach().to(torch.float32).cpu().numpy()
                        value = value_tensor.detach().to(torch.float32).cpu().numpy()

                        # Ensure contiguous float32 and correct dims
                        if policy.ndim != 2:
                            policy = policy.reshape(policy.shape[0], -1)
                        policy = np.ascontiguousarray(policy, dtype=np.float32)

                        if value.ndim == 1:
                            value = value[:, None]
                        elif value.ndim > 2:
                            value = value.reshape(value.shape[0], -1)
                        if value.shape[1] != 1:
                            # Clamp to one column if model produced extra dims
                            value = value[:, :1]
                        value = np.ascontiguousarray(value, dtype=np.float32)

                        logger.debug(f"Results converted to numpy: policy shape {policy.shape}, value shape {value.shape}")

                    except Exception as model_error:
                        logger.error(f"Model inference failed: {model_error}")
                        logger.error(f"Batch tensor shape: {batch_tensor.shape}")
                        logger.error(f"Batch tensor device: {batch_tensor.device}")
                        logger.error(f"Model device: {next(model.parameters()).device}")

                        # Return fallback values
                        batch_size = batch_tensor.shape[0]
                        policy = np.ones((batch_size, 4672), dtype=np.float32) / 4672
                        value = np.zeros((batch_size, 1), dtype=np.float32)

                # Distribute results back to workers
                start_idx = 0
                for worker_id in batch_indices:
                    if worker_id not in batch_sizes:
                        continue
                    batch_size = batch_sizes[worker_id]
                    end_idx = start_idx + batch_size
                    
                    try:
                        res = shared_memory_resources[worker_id]
                        # Convert numpy arrays to torch tensors before assignment
                        pol_np = policy[start_idx:end_idx]
                        val_np = value[start_idx:end_idx]
                        if val_np.ndim == 1:
                            val_np = val_np.reshape(-1, 1)
                        # Final guards
                        pol_np = np.ascontiguousarray(pol_np, dtype=np.float32)
                        val_np = np.ascontiguousarray(val_np, dtype=np.float32)

                        policy_slice = torch.from_numpy(pol_np)
                        value_slice = torch.from_numpy(val_np)

                        # Write into shared tensors
                        res["response_policy_tensor"][:batch_size].copy_(policy_slice)
                        res["response_value_tensor"][:batch_size].copy_(value_slice)
                        res["response_event"].set()  # Signal completion
                        logger.debug(f"Response sent to worker {worker_id}")
                    except Exception as e:
                        logger.error(f"Failed to send response to worker {worker_id}: {e}")
                        # Guarantee a safe fallback is written and event signaled to avoid deadlock
                        try:
                            fb_pol = np.ones((batch_size, policy.shape[1]), dtype=np.float32) / max(1, policy.shape[1])
                            fb_val = np.zeros((batch_size, 1), dtype=np.float32)
                            res["response_policy_tensor"][:batch_size].copy_(torch.from_numpy(fb_pol))
                            res["response_value_tensor"][:batch_size].copy_(torch.from_numpy(fb_val))
                            res["response_event"].set()
                            logger.warning(f"Fallback response sent to worker {worker_id}")
                        except Exception as fb_err:
                            logger.error(f"Failed to send fallback response to worker {worker_id}: {fb_err}")
                    
                    start_idx = end_idx

                # Log heartbeat periodically
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    heartbeat_counter += 1
                    logger.info(
                        f"Inference server heartbeat {heartbeat_counter}: "
                        f"processed {len(batch_indices)} workers, "
                        f"batch size {total_batch_size}, "
                        f"uptime {current_time - server_start_time:.1f}s"
                    )
                    last_heartbeat = current_time

            except Exception as e:
                logger.error(f"Error in batch processing: {e}", exc_info=True)
                # Clear events for failed workers to prevent deadlock
                for worker_id in batch_indices:
                    try:
                        res = shared_memory_resources[worker_id]
                        res["request_event"].clear()
                        res["response_event"].clear()
                    except Exception as clear_error:
                        logger.error(
                            f"Failed to clear events for worker {worker_id}: {clear_error}"
                        )
                continue

    except Exception as e:
        logger.error(f"Inference server error: {e}", exc_info=True)
    finally:
        logger.info("Inference server shutting down")


class InferenceClient:
    """Client using shared memory for inference."""

    def __init__(self, resources: Dict[str, Any]):
        self.res = resources
        # Get a logger instance for the client
        self.logger = logging.getLogger(__name__)

    def infer_np(self, arr_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Accept (C,H,W) or (B,C,H,W)
        if arr_batch.ndim == 3:
            arr_batch = np.expand_dims(arr_batch, 0)
        if arr_batch.ndim != 4:
            self.logger.error(f"Invalid input shape for inference: {arr_batch.shape}")
            raise ValueError("InferenceClient expects (B,C,H,W) or (C,H,W)")
        batch_size = int(arr_batch.shape[0])

        # Ensure dtype float32
        if arr_batch.dtype != np.float32:
            arr_batch = arr_batch.astype(np.float32, copy=False)

        # Adaptive timeout based on batch size and complexity
        # Chess AI needs reasonable timeouts for 53M parameter model
        base_timeout = 12.0  # 12s base timeout for 53M model
        if batch_size == 1:
            timeout = base_timeout * 1.5  # 18s for single samples (MCTS root)
        elif batch_size <= 4:
            timeout = base_timeout * 1.25  # 15s for small batches
        elif batch_size <= 16:
            timeout = base_timeout * 1.0   # 12s for medium batches (16 samples)
        else:
            timeout = base_timeout * (1.0 + batch_size / 32.0)  # Scale with batch size

        timeout = min(timeout, 30.0)  # Cap at 30s maximum for complex batches

        self.logger.debug(
            f"Inference request: batch_size={batch_size}, timeout={timeout:.1f}s"
        )

        # Copy to shared memory with error handling
        try:
            # Copy input data to shared memory
            self.res["request_tensor"][:batch_size] = torch.from_numpy(arr_batch)
            self.res["batch_size_tensor"][0] = batch_size
            
            # Signal request
            self.res["request_event"].set()
            
            # Wait for response with timeout and retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    if self.res["response_event"].wait(timeout=timeout):
                        # Response received, copy results
                        policy = self.res["response_policy_tensor"][:batch_size].numpy()
                        value = self.res["response_value_tensor"][:batch_size].numpy()
                        
                        # Clear response event for next use
                        self.res["response_event"].clear()
                        
                        # Validate outputs
                        if policy.shape[0] != batch_size or value.shape[0] != batch_size:
                            raise ValueError(f"Response shape mismatch: policy={policy.shape}, value={value.shape}, expected_batch_size={batch_size}")
                        
                        return policy, value.flatten()
                    else:
                        raise TimeoutError(f"Inference timeout after {timeout}s")
                        
                except TimeoutError:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Inference timeout (attempt {attempt + 1}/{max_retries + 1}), retrying..."
                        )
                        # Clear events and retry
                        self.res["request_event"].clear()
                        self.res["response_event"].clear()
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                    else:
                        self.logger.error(
                            f"Inference timeout after {timeout}s for batch size {batch_size} (final attempt)"
                        )
                        # CRITICAL FIX: Return fallback values instead of raising TimeoutError
                        self.logger.warning("Returning fallback values due to timeout")
                        return self._get_fallback_values(batch_size)
                except Exception as e:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Inference error (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying..."
                        )
                        time.sleep(0.1)
                        continue
                    else:
                        self.logger.error(
                            f"Inference failed after {max_retries + 1} attempts: {e}"
                        )
                        # CRITICAL FIX: Return fallback values instead of None
                        self.logger.warning("Returning fallback values after inference failure")
                        return self._get_fallback_values(batch_size)

        except Exception as e:
            self.logger.error(f"Failed to copy data to shared memory: {e}")
            # Return fallback values on critical failure
            return self._get_fallback_values(batch_size)

    def _get_fallback_values(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return safe fallback values when inference fails completely."""
        # Return uniform policy and neutral value
        policy = np.ones((batch_size, 4672), dtype=np.float32) / 4672
        value = np.zeros(batch_size, dtype=np.float32)
        return policy, value

    def _validate_events(self, events: List[Event]) -> bool:
        """Validate that all events are in a usable state."""
        try:
            for i, ev in enumerate(events):
                # Test if event can be checked without error
                _ = ev.is_set()
            return True
        except (OSError, ValueError, RuntimeError):
            return False
