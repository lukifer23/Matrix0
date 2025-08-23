from __future__ import annotations

import argparse
import logging

from azchess.logging_utils import setup_logging
from azchess.config import Config
from azchess.model import PolicyValueNet


logger = setup_logging(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser(description="Print Matrix0 model info from config")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    model = PolicyValueNet.from_config(cfg.model())
    params = model.count_parameters()
    logger.info("Model config:")
    for k, v in cfg.model().items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Total parameters: {params:,}")
    bytes_fp32 = params * 4
    bytes_fp16 = params * 2
    logger.info(f"Approx size FP32: {bytes_fp32/1e6:.1f} MB")
    logger.info(f"Approx size FP16: {bytes_fp16/1e6:.1f} MB")


if __name__ == "__main__":
    main()

