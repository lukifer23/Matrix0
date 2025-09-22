import numpy as np
import torch

from azchess.selfplay.inference import InferenceClient, setup_shared_memory_for_worker


def test_inference_client_returns_copies_between_calls():
    planes = 1
    policy_size = 4
    max_batch_size = 2
    resources = setup_shared_memory_for_worker(
        worker_id=0, planes=planes, policy_size=policy_size, max_batch_size=max_batch_size
    )
    client = InferenceClient(resources)

    first_policy = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    first_value = np.array([[0.5]], dtype=np.float32)

    resources["response_policy_tensor"][:1] = torch.from_numpy(first_policy)
    resources["response_value_tensor"][:1] = torch.from_numpy(first_value)
    resources["response_event"].set()

    request_batch = np.zeros((1, planes, 8, 8), dtype=np.float32)
    policy_first, value_first = client.infer_np(request_batch)

    np.testing.assert_allclose(policy_first, first_policy)
    np.testing.assert_allclose(value_first, first_value.flatten())

    second_policy = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    second_value = np.array([[-0.25]], dtype=np.float32)

    resources["response_policy_tensor"][:1] = torch.from_numpy(second_policy)
    resources["response_value_tensor"][:1] = torch.from_numpy(second_value)
    resources["response_event"].set()

    request_batch_2 = np.ones((1, planes, 8, 8), dtype=np.float32)
    policy_second, value_second = client.infer_np(request_batch_2)

    np.testing.assert_allclose(policy_second, second_policy)
    np.testing.assert_allclose(value_second, second_value.flatten())

    np.testing.assert_allclose(policy_first, first_policy)
    np.testing.assert_allclose(value_first, first_value.flatten())
