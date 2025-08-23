from __future__ import annotations

from torch.multiprocessing import Event
from typing import Any, Dict, List, Optional, Tuple
import time
import logging

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

        model = PolicyValueNet.from_config(model_cfg).to(device)
        logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )

        if model_state_dict:
            logger.info(
                f"Loading model from state_dict with {len(model_state_dict)} layers"
            )
            try:
                missing, unexpected = model.load_state_dict(
                    model_state_dict, strict=False
                )
                if missing:
                    logger.warning(
                        f"Missing keys during load (initialized from defaults): {len(missing)} keys"
                    )
                    logger.debug(f"Missing keys: {sorted(list(missing))[:5]}")
                if unexpected:
                    logger.warning(
                        f"Unexpected keys during load (ignored): {len(unexpected)} keys"
                    )
                    logger.debug(f"Unexpected keys: {sorted(list(unexpected))[:5]}")
                logger.info("Model loaded from state_dict successfully (non-strict).")
                # Log parameter count for clarity
                actual_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Model loaded with {actual_params:,} total parameters")
            except Exception as e:
                logger.error(
                    f"Strict load failed: {e}; attempting non-strict fallback."
                )
                model.load_state_dict(model_state_dict, strict=False)
        else:
            logger.warning("No model state_dict provided, using random weights.")
        model.eval()

        # Log device and memory info
        logger.info(f"Model device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            logger.info(
                f"CUDA memory: {torch.cuda.memory_allocated()/1024/1024:.1f}MB allocated"
            )

        server_ready_event.set()
        logger.info("Inference server ready")

        device_type = device.split(":")[0]
        use_amp = device_type in ("cuda", "mps")

        worker_events = [res["request_event"] for res in shared_memory_resources]

        logger.debug("Inference server entering main processing loop.")
        while not stop_event.is_set():
            try:
                # Wait for any worker to signal a request with efficient polling
                ready_events = []
                for i, event in enumerate(worker_events):
                    if event.is_set():
                        ready_events.append(i)

                if not ready_events:
                    # No events ready, sleep briefly to avoid busy waiting
                    time.sleep(0.001)
                    continue

                batch_indices = ready_events
                logger.debug(
                    f"Inference server processing batch from workers: {batch_indices}"
                )

            except Exception as e:
                logger.error(f"Error in inference server main loop: {e}")
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

                # Run inference with performance monitoring
                start_time = time.time()
                with torch.no_grad(), torch.autocast(
                    device_type=device_type, enabled=use_amp
                ):
                    p, v = model(batch_tensor)
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

                p_cpu = p.detach().cpu()
                v_cpu = v.detach().cpu().unsqueeze(-1)

                logger.debug(
                    f"Inference server sending responses to workers: {list(batch_sizes.keys())}"
                )
                # Write results back to shared memory efficiently
                offset = 0
                for worker_id in sorted(batch_sizes.keys()):  # Process in order
                    res = shared_memory_resources[worker_id]
                    size = batch_sizes[worker_id]
                    # Use non-blocking copies for better performance
                    res["response_policy_tensor"][:size].copy_(
                        p_cpu[offset : offset + size], non_blocking=True
                    )
                    res["response_value_tensor"][:size].copy_(
                        v_cpu[offset : offset + size], non_blocking=True
                    )
                    res["response_event"].set()  # Signal response is ready
                    logger.debug(f"Response sent to worker {worker_id}, size {size}")
                    offset += size

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

                else:
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
                    raise
