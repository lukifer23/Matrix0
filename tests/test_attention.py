import unittest

import torch

from azchess.model.resnet import NetConfig, PolicyValueNet


class TestAttentionModule(unittest.TestCase):
    def test_forward_shapes(self):
        cfg = NetConfig(
            planes=19,
            channels=64,  # divisible by 8 heads
            blocks=3,
            policy_size=4672,
            se=False,
            attention=True,
            attention_heads=8,
            chess_features=False,
            self_supervised=False,
        )
        model = PolicyValueNet(cfg)
        x = torch.randn(2, 19, 8, 8)
        p, v = model(x, return_ssl=False)
        self.assertEqual(p.shape, (2, 4672))
        self.assertEqual(v.shape, (2,))


if __name__ == '__main__':
    unittest.main()

