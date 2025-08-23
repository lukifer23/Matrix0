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
            model = PolicyValueNet.from_config(model_cfg)
            logger.info("Model created successfully, moving to device...")

            # Clear MPS cache before moving model
            if device == "mps":
                logger.info("Clearing MPS cache before model placement")
                import torch.mps
                torch.mps.empty_cache()

            # Move model to device with error handling
            model = model.to(device)
            logger.info(f"Model successfully moved to device: {device}")

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
        try:
            model.eval()
            logger.info("Model set to eval mode")
        except Exception as e:
            logger.error(f"Failed to set model to eval mode: {e}")
            raise

        # Log device and memory info
        try:
            logger.info(f"Model device: {next(model.parameters()).device}")
        except Exception as e:
            logger.error(f"Failed to get model device info: {e}")
            raise
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
        import time
        start_time = time.time()
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
                # Custom wait implementation for multiprocessing.Event compatibility
                ready_events = []
                start_time = time.time()

                while not stop_event.is_set():
                    current_time = time.time()
                    if timeout is not None and (current_time - start_time) > timeout:
                        break

                    ready_events = []
                    for i, ev in enumerate(worker_events):
                        if ev.is_set():
                            ready_events.append(ev)
                            # Clear the event for next use
                            ev.clear()

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
                        tensors_to_process.append(res["request_tensor"][:batch_size])
                        batch_sizes[worker_id] = batch_size
                        total_batch_size += batch_size
                        res["request_event"].clear()  # Clear event after reading
                        logger.debug(
                            f"Accumulating batch: worker {worker_id}, size {batch_size}, total {total_batch_size}"
                        )
                    else:
                        # Process small batches immediately to avoid timeouts
                        # Don't defer them - this was causing the 6-second delays!
                        tensors_to_process.append(res["request_tensor"][:batch_size])
                        batch_sizes[worker_id] = batch_size
                        total_batch_size += batch_size
                        res["request_event"].clear()
                        logger.debug(
                            f"Processing batch immediately: worker {worker_id}, size {batch_size}, total {total_batch_size}"
                        )

                if not tensors_to_process:
                    logger.debug("No tensors to process, continuing...")
                    continue

                # Process larger batches for better GPU utilization
                batch_tensor = torch.cat(tensors_to_process, dim=0).to(device)
                logger.debug(f"Processing batch of size {batch_tensor.shape[0]}")

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

                # Run inference with performance monitoring and error handling
                start_time = time.time()
                try:
                    with torch.no_grad(), torch.autocast(
                        device_type=device_type, enabled=use_amp
                    ):
                        p, v = model(batch_tensor)
                except Exception as inference_error:
                    logger.error(f"Model inference failed: {inference_error}")
                    logger.error(f"Batch tensor shape: {batch_tensor.shape}")
                    logger.error(f"Model device: {model.device}")
                    if device == "mps":
                        logger.info("Clearing MPS cache due to inference error")
                        torch.mps.empty_cache()
                    raise
                inference_time = time.time() - start_time

                # Log performance metrics
                # OPTIMIZED: More realistic thresholds for 32M parameter model
                if (
                    inference_time > 0.2
                ):  # Log slow inference (>200ms for larger batches)
                    logger.warning(
                        f"Slow inference: {inference_time:.3f}s for batch size {batch_tensor.shape[0]} ({inference_time/batch_tensor.shape[0]:.3f}s per sample)"
                    )
                elif inference_time > 0.1:  # Log moderate inference (100-200ms)
                    logger.info(
                        f"Moderate inference: {inference_time:.3f}s for batch size {batch_tensor.shape[0]} ({inference_time/batch_tensor.shape[0]:.3f}s per sample)"
                    )
                else:
                    logger.debug(
                        f"Fast inference: {inference_time:.3f}s for batch size {batch_tensor.shape[0]} ({inference_time/batch_tensor.shape[0]:.3f}s per sample)"
                    )

                try:
                    p_cpu = p.detach().cpu()
                    v_cpu = v.detach().cpu().unsqueeze(-1)
                except Exception as tensor_error:
                    logger.error(f"Failed to move tensors to CPU: {tensor_error}")
                    raise

                logger.debug(
                    f"Inference server sending responses to workers: {list(batch_sizes.keys())}"
                )
                # Write results back to shared memory efficiently
                offset = 0
                for worker_id in sorted(batch_sizes.keys()):  # Process in order
                    res = shared_memory_resources[worker_id]
                    size = batch_sizes[worker_id]
                    try:
                        # Use non-blocking copies for better performance
                        res["response_policy_tensor"][:size].copy_(
                            p_cpu[offset : offset + size], non_blocking=True
                        )
                        res["response_value_tensor"][:size].copy_(
                            v_cpu[offset : offset + size], non_blocking=True
                        )
                        res["response_event"].set()  # Signal response is ready
                    except Exception as copy_error:
                        logger.error(f"Failed to copy results to worker {worker_id}: {copy_error}")
                        raise
                    logger.debug(f"Response sent to worker {worker_id}, size {size}")
                    offset += size

                    # Inference server heartbeat monitoring
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        heartbeat_counter += 1
                        logger.info(f"INFERENCE_SERVER_HB: Active for {current_time - start_time:.1f}s | "
                                  f"Processed {heartbeat_counter} heartbeats | "
                                  f"Event recreations: {event_recreation_count}")
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
        # Chess AI needs reasonable timeouts for 32M parameter model
        base_timeout = 2.0  # 2s base timeout for 32M model
        if batch_size == 1:
            timeout = base_timeout * 1.5  # 3s for single samples
        elif batch_size <= 4:
            timeout = base_timeout * 1.2  # 2.4s for small batches
        else:
            timeout = base_timeout * (1.0 + batch_size / 32.0)  # Scale with batch size

        timeout = min(timeout, 5.0)  # Cap at 5s maximum for 32M model

        self.logger.debug(
            f"Inference request: batch_size={batch_size}, timeout={timeout:.1f}s"
        )

        # Copy to shared memory with error handling
        try:
            # CRITICAL FIX: Set the batch size tensor so the server knows how much data to process
            self.res["batch_size_tensor"][0] = batch_size
            self.logger.debug(f"Set batch_size_tensor to {batch_size}")

            self.res["request_tensor"][:batch_size].copy_(torch.from_numpy(arr_batch))
            self.logger.debug(f"Copied {batch_size} samples to request_tensor")

            self.res["request_event"].set()
            self.logger.debug(f"Set request_event for batch size {batch_size}")
        except Exception as e:
            # Handle potential event corruption issues
            if "Invalid file object" in str(e):
                self.logger.warning("Request event may be corrupted, attempting recovery")
                try:
                    # Try to recreate the event
                    self.res["request_event"] = Event()
                    self.res["response_event"] = Event()
                    self.logger.info("Recreated corrupted events, retrying request")

                    # Retry the request with new events
                    self.res["batch_size_tensor"][0] = batch_size
                    self.res["request_tensor"][:batch_size].copy_(torch.from_numpy(arr_batch))
                    self.res["request_event"].set()
                    self.logger.debug("Successfully retried request with new events")
                except Exception as recovery_error:
                    self.logger.error(f"Failed to recover from event corruption: {recovery_error}")
                    raise RuntimeError(f"Inference request failed after event recovery: {recovery_error}")
            else:
                self.logger.error(f"Failed to copy request to shared memory: {e}")
                raise RuntimeError(f"Inference request failed: {e}")

        # Wait for response with timeout and retry logic
        max_retries = 1  # Reduced from 2 to prevent cascading failures
        for attempt in range(max_retries + 1):
            try:
                if self.res["response_event"].wait(timeout=timeout):
                    # Read response from shared memory
                    policy = self.res["response_policy_tensor"][:batch_size].numpy()
                    value = (
                        self.res["response_value_tensor"][:batch_size].numpy().flatten()
                    )

                    # Validate response
                    if policy.shape[0] != batch_size or value.shape[0] != batch_size:
                        raise ValueError(
                            f"Response shape mismatch: policy={policy.shape}, value={value.shape}, expected={batch_size}"
                        )

                    self.res["response_event"].clear()
                    return policy, value
            except (ValueError, RuntimeError) as e:
                # Handle event corruption during wait
                if "Invalid file object" in str(e):
                    self.logger.warning(f"Response event corrupted during wait (attempt {attempt + 1})")
                    if attempt < max_retries:
                        # Try to recreate events and retry
                        try:
                            self.res["request_event"] = Event()
                            self.res["response_event"] = Event()
                            # Resend the request
                            self.res["batch_size_tensor"][0] = batch_size
                            self.res["request_tensor"][:batch_size].copy_(torch.from_numpy(arr_batch))
                            self.res["request_event"].set()
                            continue
                        except Exception as recovery_error:
                            self.logger.error(f"Failed to recover from response event corruption: {recovery_error}")
                            if attempt == max_retries:
                                raise RuntimeError(f"Inference failed after {max_retries + 1} attempts due to event corruption: {recovery_error}")
                            continue
                    else:
                        raise RuntimeError(f"Inference failed after {max_retries + 1} attempts due to event corruption: {e}")
                else:
                    raise

            except Exception as timeout_e:
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
                    raise TimeoutError(
                        f"Inference timeout after {timeout}s for batch size {batch_size}"
                    )

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
                    # CRITICAL: Return None explicitly to prevent unpacking None in MCTS
                    return None
