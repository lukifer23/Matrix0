from __future__ import annotations

import os
import platform


def main():
    try:
        import torch
        print(f"Python: {platform.python_version()} arch={platform.machine()}")
        print(f"Torch: {torch.__version__}")
        built = getattr(torch.backends.mps, 'is_built', lambda: False)()
        avail = getattr(torch.backends.mps, 'is_available', lambda: False)()
        print(f"MPS built: {built} available: {avail}")
        if avail:
            import torch.mps  # noqa: F401
            print("MPS module import: OK")
        print(f"select_device(auto): ", end='')
        from ..config import select_device
        print(select_device('auto'))
        print("Env: PYTORCH_ENABLE_MPS_FALLBACK=", os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))
        print("Env: PYTORCH_MPS_HIGH_WATERMARK_RATIO=", os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'))
        if avail:
            x = torch.randn(4, 19, 8, 8, device='mps')
            print("Allocated test tensor on mps: ", x.shape)
    except Exception as e:
        print("Diag failed:", e)

if __name__ == '__main__':
    main()

