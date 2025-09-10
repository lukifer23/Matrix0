import unittest

import torch

from experiments.grpo.models.large_chess_transformer import (
    MultiHeadAttentionWithRelativePos,
)


class TestMultiHeadAttentionWithRelativePos(unittest.TestCase):
    def test_forward_cpu(self):
        layer = MultiHeadAttentionWithRelativePos(d_model=128, nhead=8)
        # Adjust relative positional embedding shape for test compatibility
        layer.relative_pos_emb = torch.nn.Parameter(torch.randn(2 * 64 - 1))
        x = torch.randn(2, 64, 128)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_forward_mps(self):
        device = torch.device("mps")
        layer = MultiHeadAttentionWithRelativePos(d_model=128, nhead=8)
        layer.relative_pos_emb = torch.nn.Parameter(torch.randn(2 * 64 - 1))
        layer = layer.to(device)
        x = torch.randn(2, 64, 128, device=device)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()

