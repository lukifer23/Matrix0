from __future__ import annotations

import uuid
from torch.multiprocessing import Process, Event, Queue
from typing import Any, Dict, Tuple, List
import os
import time
import queue
import logging

import numpy as np
import torch

from ..model import PolicyValueNet

# --- Shared Memory IPC Implementation ---

def setup_shared_memory_for_worker(worker_id: int, planes: int, policy_size: int, max_batch_size: int) -> Dict[str, Any]:
    """Creates shared memory tensors and events for a single worker."""
    return {
        'request_tensor': torch.zeros((max_batch_size, planes, 8, 8), dtype=torch.float32).share_memory_(),
        'response_policy_tensor': torch.zeros((max_batch_size, policy_size), dtype=torch.float32).share_memory_(),
        'response_value_tensor': torch.zeros((max_batch_size, 1), dtype=torch.float32).share_memory_(),
        'request_event': Event(),
        'response_event': Event(),
        'batch_size_tensor': torch.tensor([0], dtype=torch.int32).share_memory_(),
    }

def run_inference_server(
    device: str, 
    model_cfg: dict, 
    model_state_dict: Dict[str, Any] | None, 
    stop_event: Event, 
    server_ready_event: Event,
    shared_memory_resources: List[Dict[str, Any]]
):
    """Inference server using shared memory for communication."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Inference server starting on device: {device}")
        model = PolicyValueNet.from_config(model_cfg).to(device)
        if model_state_dict:
            logger.info(f"Loading model from state_dict with {len(model_state_dict)} parameters")
            model.load_state_dict(model_state_dict)
            logger.info(f"Model loaded from state_dict successfully.")
        else:
            logger.warning("No model state_dict provided, using random weights.")
        model.eval()
        
        server_ready_event.set()
        logger.info("Inference server ready")

        device_type = device.split(':')[0]
        use_amp = device_type in ("cuda", "mps")
        
        worker_events = [res['request_event'] for res in shared_memory_resources]

        logger.debug("Inference server entering main processing loop.")
        while not stop_event.is_set():
            # Wait for any worker to signal a request
            ready_workers = [event.wait(0.1) for event in worker_events] # Short timeout to check stop_event
            
            batch_indices = [i for i, is_ready in enumerate(ready_workers) if is_ready]
            if not batch_indices:
                continue
            
            logger.debug(f"Inference server received requests from workers: {batch_indices}")

            # Prepare batch from shared memory
            tensors_to_process = []
            batch_sizes = {}
            for worker_id in batch_indices:
                res = shared_memory_resources[worker_id]
                batch_size = res['batch_size_tensor'].item()
                if batch_size <= 0:
                    # Spurious wakeup or protocol error; ignore
                    continue
                tensors_to_process.append(res['request_tensor'][:batch_size])
                batch_sizes[worker_id] = batch_size
                res['request_event'].clear() # Clear event after reading

            if not tensors_to_process:
                continue

            batch_tensor = torch.cat(tensors_to_process, dim=0).to(device)

            # Run inference
            with torch.no_grad(), torch.autocast(device_type=device_type, enabled=use_amp):
                p, v = model(batch_tensor)
            
            p_cpu = p.detach().cpu()
            v_cpu = v.detach().cpu().unsqueeze(-1)

            logger.debug(f"Inference server sending responses to workers: {batch_indices}")
            # Write results back to shared memory
            offset = 0
            for worker_id in batch_indices:
                res = shared_memory_resources[worker_id]
                size = batch_sizes[worker_id]
                res['response_policy_tensor'][:size].copy_(p_cpu[offset:offset+size])
                res['response_value_tensor'][:size].copy_(v_cpu[offset:offset+size])
                res['response_event'].set() # Signal response is ready
                offset += size

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

        # Write data to shared memory
        self.res['batch_size_tensor'][0] = batch_size
        self.res['request_tensor'][:batch_size].copy_(torch.from_numpy(arr_batch))
        
        # Signal server
        self.res['response_event'].clear()
        self.logger.debug(f"Client sending {batch_size} inference requests.")
        self.res['request_event'].set()
        
        # Wait for response
        if not self.res['response_event'].wait(timeout=10.0):
            self.logger.error("Inference request timed out after 10 seconds.")
            raise TimeoutError("Inference timeout after 10 seconds")
            
        self.logger.debug("Client received inference response.")
        # Read response from shared memory
        policy = self.res['response_policy_tensor'][:batch_size].numpy()
        value = self.res['response_value_tensor'][:batch_size].numpy().flatten()
        
        return policy, value
