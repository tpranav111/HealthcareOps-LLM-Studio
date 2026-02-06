import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.config import load_config
from src.training.distill import distill_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    distill_model(config)


if __name__ == "__main__":
    main()
